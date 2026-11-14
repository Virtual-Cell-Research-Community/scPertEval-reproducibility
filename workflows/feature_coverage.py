"""Audit model-feature coverage for the K562 Essential perturbation cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import anndata as ad
import pandas as pd

SOURCE_PROVENANCE = {
    "canonical_input_sha256": "7e9a4ab48f60de0b7cacb88224dad0145347f95a86a68db9cddfbb8a9c50c216",
    "gears_gene2go_url": "https://dataverse.harvard.edu/api/access/datafile/6153417",
    "genept_archive_url": "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1",
    "genept_archive_sha256": "6193575dcbd7bbf214c8ca3eb518cb3d13272443f98ecd6c26402411a7ca745e",
    "genept_member": "GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.",
    "miller_repository_revision": "15361aec8392fb45a75c8c9da88ca5e6510fca97",
    "ensembl_canonical_url": "https://ftp.ensembl.org/pub/release-113/tsv/homo_sapiens/Homo_sapiens.GRCh38.113.canonical.tsv.gz",
    "ensembl_canonical_sha256": "df52a35bd8e6bfbf92a96f56e6847b7eebf12a056b4253dd3e1087b535469764",
    "ensembl_peptide_url": "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz",
    "ensembl_peptide_sha256": "c51944dc7e44a72d7370d208a24a038d5a4672f8d2af32b5ea9df039b8e2ffd5",
    "esm2_model": "facebook/esm2_t48_15B_UR50D",
    "esm2_model_revision": "5fbca39631164edc1d402a5aa369f982f72ee282",
    "esm2_config_url": "https://huggingface.co/facebook/esm2_t48_15B_UR50D/resolve/5fbca39631164edc1d402a5aa369f982f72ee282/config.json",
    "esm2_config_sha256": "4e5c41d32d54e79ade1a12acb22cac8b9a189d46676f8fe9674e342432f000ba",
    "esm2_max_position_embeddings": 1026,
    "esm2_max_protein_residues": 1024,
    "txpert_repository_revision": "08d82eea86746b044cf7531f4ec8c5f60e1cb73f",
    "txpert_gene_set_path": "data/gears_gene_set.csv",
    "txpert_gene_set_sha256": "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d",
}

ESM2_MAX_PROTEIN_RESIDUES = 1024  # 1026 positions minus BOS/EOS special tokens


def _keyset_sha256(keys: set[str]) -> str:
    payload = json.dumps(sorted(keys), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _reference_keys(path: Path) -> set[str]:
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise TypeError(f"expected a keyed JSON reference object in {path}")
        return set(map(str, value))
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not hasattr(value, "keys"):
        raise TypeError(f"expected a keyed embedding/reference object in {path}")
    return set(map(str, value.keys()))


def _txpert_keys(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    if "0" not in frame:
        raise ValueError(f"TxPert gene set {path} has no '0' gene-symbol column")
    return set(frame["0"].dropna().astype(str))


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _miller_esm2_keys(canonical_path: Path, peptide_fasta_path: Path) -> set[str]:
    """Genes for which Miller's pinned ESM2 preparation can generate an embedding."""
    canonical = pd.read_csv(
        canonical_path,
        sep="\t",
        header=None,
        names=["gene_id", "transcript_id", "status"],
        compression="infer",
    )
    transcripts = set(
        canonical.loc[canonical["status"] == "Ensembl Canonical", "transcript_id"].astype(str)
    )

    # Miller sorts canonical sequences by descending length, then drops duplicate gene
    # symbols while keeping the first row. Reproduce that exactly: a gene is eligible only
    # when its longest canonical peptide fits ESM2's positional table.
    maximum_length_by_gene: dict[str, int] = {}
    transcript: str | None = None
    gene_symbol: str | None = None
    sequence_length = 0

    def retain_record() -> None:
        if transcript in transcripts and gene_symbol and gene_symbol != "nan":
            maximum_length_by_gene[gene_symbol] = max(
                sequence_length,
                maximum_length_by_gene.get(gene_symbol, 0),
            )

    with _open_text(peptide_fasta_path) as handle:
        for line in handle:
            if line.startswith(">"):
                retain_record()
                fields = {
                    key: value
                    for item in line[1:].split()
                    if ":" in item
                    for key, value in [item.split(":", 1)]
                }
                transcript = fields.get("transcript")
                gene_symbol = fields.get("gene_symbol")
                sequence_length = 0
            else:
                sequence_length += len(line.strip())
    retain_record()
    return {
        gene
        for gene, sequence_length in maximum_length_by_gene.items()
        if sequence_length <= ESM2_MAX_PROTEIN_RESIDUES
    }


def audit_feature_coverage(
    *,
    input_path: Path,
    gene2go_path: Path,
    genept_path: Path,
    esm2_canonical_path: Path,
    esm2_peptide_fasta_path: Path,
    txpert_gene_set_path: Path,
    minimum_cells: int = 50,
) -> dict:
    """Return the exact common target cohort and coverage provenance."""
    data = ad.read_h5ad(input_path, backed="r")
    try:
        required = {"perturbation", "is_control"}
        missing_fields = sorted(required - set(data.obs.columns))
        if missing_fields:
            raise ValueError(f"canonical input is missing obs fields: {missing_fields}")

        labels = data.obs["perturbation"].astype("string")
        controls = data.obs["is_control"]
        if controls.isna().any() or not set(controls.unique()) <= {True, False}:
            raise ValueError("obs['is_control'] must be a complete boolean column")
        counts = labels[~controls.astype(bool)].value_counts()
        n_control_cells = int(controls.astype(bool).sum())
        at_minimum = set(map(str, counts[counts >= minimum_cells].index))
        measured = at_minimum & set(map(str, data.var_names))
        shape = [int(data.n_obs), int(data.n_vars)]
    finally:
        data.file.close()

    support = {
        "gears_gene2go": _reference_keys(gene2go_path),
        "genept": _reference_keys(genept_path),
        "miller_esm2": _miller_esm2_keys(esm2_canonical_path, esm2_peptide_fasta_path),
        "txpert_public": _txpert_keys(txpert_gene_set_path),
    }
    preliminary = measured & support["gears_gene2go"] & support["genept"]
    final = preliminary & support["miller_esm2"] & support["txpert_public"]
    final_counts = counts.reindex(sorted(final))

    return {
        "schema_version": 1,
        "dataset": "replogle-weissman-2022-k562-essential",
        "sources": SOURCE_PROVENANCE,
        "canonical_shape": shape,
        "minimum_treated_cells_per_perturbation": minimum_cells,
        "counts": {
            "source_targets": int(counts.size),
            "targets_at_minimum": len(at_minimum),
            "measured_targets_at_minimum": len(measured),
            "preliminary_gears_genept_common": len(preliminary),
            "esm2_supported_in_preliminary": len(preliminary & support["miller_esm2"]),
            "txpert_public_supported_in_preliminary": len(
                preliminary & support["txpert_public"]
            ),
            "final_common_targets": len(final),
            "final_targeted_cells": int(final_counts.sum()),
            "control_cells": n_control_cells,
            "minimum_final_target_cells": int(final_counts.min()),
            "median_final_target_cells": float(final_counts.median()),
            "maximum_final_target_cells": int(final_counts.max()),
        },
        "retention_fraction_of_preliminary": len(final) / len(preliminary),
        "support_keysets": {
            name: {"n_keys": len(keys), "sha256": _keyset_sha256(keys)}
            for name, keys in support.items()
        },
        "missing_from_esm2": sorted(preliminary - support["miller_esm2"]),
        "missing_from_txpert_public": sorted(preliminary - support["txpert_public"]),
        "final_targets": sorted(final),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gene2go", type=Path, required=True)
    parser.add_argument("--genept", type=Path, required=True)
    parser.add_argument("--esm2-canonical", type=Path, required=True)
    parser.add_argument("--esm2-peptide-fasta", type=Path, required=True)
    parser.add_argument("--txpert-gene-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cells", type=int, default=50)
    parser.add_argument("--expected-preliminary", type=int, default=1574)
    args = parser.parse_args()

    report = audit_feature_coverage(
        input_path=args.input,
        gene2go_path=args.gene2go,
        genept_path=args.genept,
        esm2_canonical_path=args.esm2_canonical,
        esm2_peptide_fasta_path=args.esm2_peptide_fasta,
        txpert_gene_set_path=args.txpert_gene_set,
        minimum_cells=args.minimum_cells,
    )
    observed = report["counts"]["preliminary_gears_genept_common"]
    if observed != args.expected_preliminary:
        raise ValueError(
            f"preliminary cohort changed: observed={observed}, expected={args.expected_preliminary}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report["counts"], "retention": report["retention_fraction_of_preliminary"]}, indent=2))


if __name__ == "__main__":
    main()
