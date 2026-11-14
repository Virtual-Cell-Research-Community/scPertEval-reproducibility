"""Generate the pinned ESM2 target-gene embedding asset used by fMLP-ESM2.

This preserves Miller et al.'s sequence selection and mean-pooling semantics while making
the expensive inference restart-safe. Each gene is cached atomically before the next gene
starts; the final compact dictionary and its provenance manifest are also written atomically.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

PINNED_MODEL = "facebook/esm2_t48_15B_UR50D"
PINNED_REVISION = "5fbca39631164edc1d402a5aa369f982f72ee282"
PINNED_HIDDEN_SIZE = 5120
PINNED_MAX_RESIDUES = 1024
IMPLEMENTATION_REVISION = (
    "shiftbioscience/Perturbation-Models-Outperform-Baselines@"
    "15361aec8392fb45a75c8c9da88ca5e6510fca97"
)


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def target_sha256(targets: list[str]) -> str:
    payload = json.dumps(sorted(targets), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _fasta_records(path: Path) -> dict[str, tuple[str | None, str]]:
    """Return transcript -> (gene symbol, peptide) from Ensembl's peptide FASTA."""
    records: dict[str, tuple[str | None, str]] = {}
    transcript: str | None = None
    gene_symbol: str | None = None
    sequence: list[str] = []

    def retain() -> None:
        if transcript is not None:
            records[transcript] = (gene_symbol, "".join(sequence))

    with _open_text(path) as stream:
        for line in stream:
            if line.startswith(">"):
                retain()
                fields = {
                    key: value
                    for item in line[1:].split()
                    if ":" in item
                    for key, value in [item.split(":", 1)]
                }
                transcript = fields.get("transcript")
                gene_symbol = fields.get("gene_symbol")
                sequence = []
            else:
                sequence.append(line.strip())
    retain()
    return records


def select_miller_sequences(
    canonical_path: Path,
    peptide_fasta_path: Path,
) -> dict[str, str]:
    """Reproduce Miller's longest-canonical-peptide selection per gene symbol."""
    canonical = pd.read_csv(
        canonical_path,
        sep="\t",
        header=None,
        names=["gene_id", "transcript_id", "status"],
        compression="infer",
    )
    canonical = canonical.loc[canonical["status"] == "Ensembl Canonical"]
    fasta = _fasta_records(peptide_fasta_path)
    rows = []
    for transcript in canonical["transcript_id"].astype(str):
        gene_symbol, sequence = fasta.get(transcript, (None, ""))
        if gene_symbol and sequence:
            rows.append({"gene_symbol": gene_symbol, "sequence": sequence, "len_seq": len(sequence)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no canonical Ensembl peptide records could be aligned")
    frame = frame.sort_values("len_seq", ascending=False).drop_duplicates("gene_symbol")
    return dict(zip(frame["gene_symbol"].astype(str), frame["sequence"].astype(str), strict=True))


def _atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _cache_path(cache_dir: Path, gene: str) -> Path:
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", gene)[:80]
    identity = hashlib.sha256(gene.encode()).hexdigest()[:16]
    return cache_dir / f"{readable}--{identity}.npy"


def _load_cached(path: Path, *, expected_dimension: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if value.shape != (expected_dimension,) or not np.isfinite(value).all():
        return None
    return value.astype(np.float32, copy=False)


def _atomic_numpy(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def generate_embeddings(args: argparse.Namespace) -> dict:
    experiment = json.loads(args.experiment.read_text())
    targets = sorted(map(str, experiment["perturbations"]))
    if len(targets) != int(experiment["n_perturbations"]):
        raise ValueError("experiment perturbation count does not match its target list")
    if (args.model, args.revision) != (PINNED_MODEL, PINNED_REVISION) and not args.smoke:
        raise ValueError("an unpinned checkpoint is permitted only with --smoke")

    selected = select_miller_sequences(args.canonical, args.peptide_fasta)
    missing = sorted(set(targets) - set(selected))
    overlength = sorted(gene for gene in targets if gene in selected and len(selected[gene]) > PINNED_MAX_RESIDUES)
    if missing or overlength:
        raise ValueError(
            f"frozen target coverage changed: missing={len(missing)}, overlength={len(overlength)}"
        )
    selected_targets = targets[: args.smoke] if args.smoke else targets

    import torch
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.huggingface_cache,
    )
    model = EsmModel.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.huggingface_cache,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model.eval()
    dimension = int(model.config.hidden_size)
    max_positions = int(model.config.max_position_embeddings)
    if not args.smoke and dimension != PINNED_HIDDEN_SIZE:
        raise ValueError(f"pinned ESM2 hidden size changed: observed={dimension}, expected={PINNED_HIDDEN_SIZE}")
    if not args.smoke and max_positions != PINNED_MAX_RESIDUES + 2:
        raise ValueError(
            "pinned ESM2 positional table changed: "
            f"observed={max_positions}, expected={PINNED_MAX_RESIDUES + 2}"
        )

    revision_key = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{args.model}@{args.revision}")
    cache_dir = args.cache_dir / revision_key / "mean-including-special-tokens"
    embeddings: dict[str, np.ndarray] = {}
    cache_hits = 0
    started = time.monotonic()
    for index, gene in enumerate(selected_targets, start=1):
        path = _cache_path(cache_dir, gene)
        embedding = _load_cached(path, expected_dimension=dimension)
        if embedding is None:
            tokens = tokenizer(selected[gene], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
                embedding = hidden.mean(dim=1)[0].float().cpu().numpy()
            if embedding.shape != (dimension,) or not np.isfinite(embedding).all():
                raise ValueError(f"invalid embedding generated for {gene}: shape={embedding.shape}")
            _atomic_numpy(embedding, path)
        else:
            cache_hits += 1
        embeddings[gene] = embedding.astype(np.float32, copy=False)
        elapsed = time.monotonic() - started
        print(
            f"[{index}/{len(selected_targets)}] {gene}: residues={len(selected[gene])}, "
            f"cache_hits={cache_hits}, elapsed_seconds={elapsed:.1f}",
            flush=True,
        )

    _atomic_pickle(embeddings, args.output)
    manifest = {
        "schema_version": 1,
        "complete": not bool(args.smoke),
        "implementation_revision": IMPLEMENTATION_REVISION,
        "experiment": str(args.experiment),
        "experiment_sha256": sha256_file(args.experiment),
        "target_count": len(selected_targets),
        "frozen_target_count": len(targets),
        "target_sha256": target_sha256(selected_targets),
        "model": args.model,
        "model_revision": args.revision,
        "hidden_size": dimension,
        "max_position_embeddings": max_positions,
        "pooling": "mean over last hidden state including BOS and EOS tokens",
        "dtype": "float32",
        "sequence_selection": "longest Ensembl release 113 canonical peptide per gene symbol",
        "maximum_protein_residues": PINNED_MAX_RESIDUES,
        "canonical_path": str(args.canonical),
        "canonical_sha256": sha256_file(args.canonical),
        "peptide_fasta_path": str(args.peptide_fasta),
        "peptide_fasta_sha256": sha256_file(args.peptide_fasta),
        "cache_dir": str(cache_dir),
        "cache_hits": cache_hits,
        "elapsed_seconds": time.monotonic() - started,
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "package_versions": _package_versions(["torch", "transformers", "accelerate", "safetensors"]),
    }
    _atomic_json(manifest, args.manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--peptide-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--huggingface-cache", type=Path, required=True)
    parser.add_argument("--model", default=PINNED_MODEL)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument(
        "--smoke",
        type=int,
        metavar="N",
        help="generate only the first N targets; required when overriding the pinned model",
    )
    args = parser.parse_args()
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be a positive target count")
    manifest = generate_embeddings(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
