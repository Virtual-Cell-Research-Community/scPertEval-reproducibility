"""Train Miller et al.'s fMLP with either GenePT or ESM2 gene embeddings.

The network, optimizer, scheduler, cell-level MSE objective, validation cadence, and published
hyperparameters follow the pinned CellSimBench implementation. This adapter changes only the
data plumbing: it reads the frozen split directly, avoids materializing a repeated embedding
for every cell, never reads test-response expression during fitting, and writes the shared
resumable prediction bundle.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

if hasattr(ad, "settings"):
    ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from workflows.model_runs import add_point_run_arguments, begin_point_model_run

DATA_DIR = HERE.parent / "data"
GENEPT_PICKLE = (
    HERE.parent / "sclambda" / "cache" / "GenePT_emebdding_v2" / "GenePT_gene_protein_embedding_model_3_text.pickle."
)
ESM2_PICKLE = HERE / "cache" / "esm2-15b" / "esm2_t48_15B_UR50D_mean.pickle"
IMPLEMENTATION_REVISION = (
    "shiftbioscience/Perturbation-Models-Outperform-Baselines@15361aec8392fb45a75c8c9da88ca5e6510fca97"
)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.activation(self.fc_input(values))
        values = self.activation(self.fc_input2(values))
        return self.fc_output(values)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.activation(self.fc_hidden(values))
        values = self.activation(self.fc_hidden2(values))
        return self.fc_output(values)


class FMLPNet(nn.Module):
    """Pinned CellSimBench fMLP architecture for single-gene perturbations."""

    def __init__(self, expression_dim: int, embedding_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder_g = Encoder(embedding_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, expression_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder_g(embeddings))


class PerturbationCellDataset(Dataset):
    """Cell targets plus compact indices into one embedding table."""

    def __init__(self, expression: np.ndarray, embedding_ids: np.ndarray, embeddings: np.ndarray):
        self.expression = torch.from_numpy(expression)
        self.embedding_ids = torch.from_numpy(embedding_ids.astype(np.int64, copy=False))
        self.embeddings = torch.from_numpy(embeddings)

    def __len__(self) -> int:
        return len(self.expression)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[self.embedding_ids[index]], self.expression[index]


@dataclass(frozen=True)
class SelectedExpression:
    values: np.ndarray
    labels: np.ndarray


def load_embedding_dictionary(path: Path) -> dict[str, np.ndarray]:
    """Load a gene-indexed dictionary/DataFrame and enforce one finite shared dimension."""
    with path.open("rb") as stream:
        loaded = pickle.load(stream)
    if isinstance(loaded, pd.DataFrame):
        embeddings = {str(name): row.to_numpy(dtype=np.float32) for name, row in loaded.iterrows()}
    elif isinstance(loaded, dict):
        embeddings = {str(name): np.asarray(value, dtype=np.float32) for name, value in loaded.items()}
    else:
        raise TypeError(f"unsupported embedding artifact type: {type(loaded).__name__}")
    dimensions = {value.size for value in embeddings.values()}
    if len(dimensions) != 1 or not dimensions:
        raise ValueError("embedding artifact does not have one non-empty shared dimension")
    invalid = [name for name, value in embeddings.items() if value.ndim != 1 or not np.isfinite(value).all()]
    if invalid:
        raise ValueError(f"embedding artifact contains invalid rows: {invalid[:20]}")
    return embeddings


def embedding_table(
    embeddings: dict[str, np.ndarray],
    perturbations: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    missing = sorted(set(perturbations) - set(embeddings))
    if missing:
        raise ValueError(f"embeddings are missing perturbations: {missing[:20]}")
    ordered = sorted(set(perturbations))
    return (
        np.vstack([embeddings[name] for name in ordered]).astype(np.float32, copy=False),
        {name: index for index, name in enumerate(ordered)},
    )


def _dense_float32(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def selected_expression(
    adata: ad.AnnData,
    conditions: list[str],
    *,
    condition_key: str = "condition",
) -> SelectedExpression:
    """Read only requested expression rows from a possibly backed model view."""
    labels = adata.obs[condition_key].astype(str).to_numpy()
    selected = np.flatnonzero(np.isin(labels, conditions))
    if not len(selected):
        raise ValueError("no cells matched the requested conditions")
    subset = adata[selected].to_memory() if adata.isbacked else adata[selected].copy()
    observed = subset.obs[condition_key].astype(str).to_numpy()
    missing = sorted(set(conditions) - set(observed))
    if missing:
        raise ValueError(f"model view contains no cells for conditions: {missing[:20]}")
    return SelectedExpression(_dense_float32(subset.X), observed)


def _bare(condition: str) -> str:
    return condition.removesuffix("+ctrl")


def _loader(
    selected: SelectedExpression,
    table: np.ndarray,
    table_index: dict[str, int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ids = np.asarray([table_index[_bare(label)] for label in selected.labels], dtype=np.int64)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PerturbationCellDataset(selected.values, ids, table),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _mean_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for embeddings, expression in loader:
            embeddings = embeddings.to(device)
            expression = expression.to(device)
            prediction = model(embeddings)
            losses.append(float(nn.functional.mse_loss(prediction, expression).cpu()))
    return float(np.mean(losses))


def train_fmlp(
    train: SelectedExpression,
    validation: SelectedExpression,
    table: np.ndarray,
    table_index: dict[str, int],
    *,
    expression_dim: int,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_every: int,
    seed: int,
    device: torch.device,
) -> tuple[FMLPNet, list[dict[str, float | int | None]]]:
    """Fit the pinned fMLP and select the lowest validation-loss checkpoint."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    model = FMLPNet(expression_dim, table.shape[1], latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
    train_loader = _loader(
        train,
        table,
        table_index,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation,
        table,
        table_index,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    metrics: list[dict[str, float | int | None]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for embeddings, expression in train_loader:
            embeddings = embeddings.to(device)
            expression = expression.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(embeddings), expression)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        check_validation = epoch % val_every == 0 or epoch == epochs
        validation_loss = _mean_loss(model, validation_loader, device) if check_validation else None
        if validation_loss is not None and validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "validation_loss": validation_loss,
            }
        )
        scheduler.step()
    model.load_state_dict(best_state)
    model.eval()
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fmlp-genept", "fmlp-esm2"], required=True)
    parser.add_argument("--dataset", default="replogle-weissman-2022-k562-essential")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    add_point_run_arguments(parser)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_every, args.latent_dim, args.hidden_dim) < 1:
        raise ValueError("epochs, batch size, validation cadence, and model dimensions must be positive")

    embedding_path = args.embeddings or (GENEPT_PICKLE if args.model == "fmlp-genept" else ESM2_PICKLE)
    if not embedding_path.is_file():
        raise FileNotFoundError(f"embedding artifact is missing: {embedding_path}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device != "cpu" and device.type != "cuda":
        raise RuntimeError("full fMLP training requires a CUDA GPU; use --device cpu only for smoke tests")

    run = begin_point_model_run(
        args,
        repository_root=REPOSITORY_ROOT,
        model_id=args.model,
        environment="models/pixi.lock#fmlp",
        configuration={
            "architecture": "CellSimBench-fMLP",
            "embedding_kind": args.model.removeprefix("fmlp-"),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "validation_every_epochs": args.val_every,
            "checkpoint_selection": "minimum-cell-weighted-validation-mse",
            "device": str(device),
        },
        additional_inputs={"perturbation_embeddings": embedding_path},
    )
    if run.skipped:
        print(f"skipping verified completed run: {run.handle.directory}")
        return 0

    fold_path = DATA_DIR / args.dataset / "folds" / f"fold_{args.fold}.pkl"
    with fold_path.open("rb") as stream:
        fold = pickle.load(stream)
    train_conditions = sorted(condition for condition in fold["train"] if condition != "ctrl")
    validation_conditions = sorted(fold["val"])
    test = sorted(_bare(condition) for condition in fold["test"])

    embeddings = load_embedding_dictionary(embedding_path)
    all_perturbations = [_bare(condition) for condition in [*train_conditions, *validation_conditions]] + test
    table, table_index = embedding_table(embeddings, all_perturbations)

    adata = ad.read_h5ad(run.raw_path, backed="r")
    try:
        train = selected_expression(adata, train_conditions)
        validation = selected_expression(adata, validation_conditions)
        genes = pd.Index(np.asarray(adata.var_names).astype(str))
    finally:
        adata.file.close()

    model, metrics = train_fmlp(
        train,
        validation,
        table,
        table_index,
        expression_dim=len(genes),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_every=args.val_every,
        seed=args.seed,
        device=device,
    )
    test_embeddings = torch.from_numpy(np.vstack([embeddings[name] for name in test]).astype(np.float32)).to(device)
    with torch.no_grad():
        predicted = model(test_embeddings).cpu().numpy().astype(np.float32)

    prediction = ad.AnnData(
        X=predicted,
        obs=pd.DataFrame({"perturbation": test}, index=[f"{args.model}::{name}" for name in test]),
    )
    prediction.var_names = genes
    run.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(run.prediction_path, compression="gzip")
    run.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(run.metrics_path, index=False)

    checkpoint = run.checkpoint_dir / "model.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "configuration": json.loads((run.handle.directory / "resolved-run.json").read_text())["configuration"],
            "genes": list(genes),
        },
        checkpoint,
    )
    run.complete(implementation_revision=IMPLEMENTATION_REVISION, checkpoints=[checkpoint])
    print(f"wrote {prediction.shape} {args.model} predictions to {run.prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
