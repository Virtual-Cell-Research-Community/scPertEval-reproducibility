"""Arithmetic-preserving initialization for scLAMBDA on genome-wide cell sets.

Upstream ``Model.__init__`` scans every cell once for every condition and allocates one
``n_cells x embedding_dim`` array per target position even when ``use_tg_coord=False``.
That is harmless for the small demo datasets, but quadratic in condition count and needlessly
allocates hundreds of GiB for GWPS.  This module constructs the same tensors by categorical
indexing, omits target-coordinate arrays that the selected model never reads, and computes only
the validation condition means required by ``Model.train``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_condition_embedding_table(
    conditions: np.ndarray,
    gene_embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted conditions, per-cell table indices, and summed embeddings."""
    unique, inverse = np.unique(np.asarray(conditions, dtype=str), return_inverse=True)
    rows = []
    for condition in unique:
        genes = condition.split("+")
        value = np.asarray(gene_embeddings[genes[0]], dtype=np.float32).copy()
        for gene in genes[1:]:
            value += np.asarray(gene_embeddings[gene], dtype=np.float32)
        rows.append(value)
    return unique, inverse, np.stack(rows).astype(np.float32, copy=False)


def selected_centered_condition_means(
    matrix,
    condition_indices: np.ndarray,
    unique_conditions: np.ndarray,
    selected_conditions: np.ndarray,
    control_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute centered expression means without repeatedly scanning all condition labels."""
    order = np.argsort(condition_indices, kind="stable")
    counts = np.bincount(condition_indices, minlength=len(unique_conditions))
    stops = np.cumsum(counts)
    starts = np.concatenate(([0], stops[:-1]))
    condition_to_index = {condition: index for index, condition in enumerate(unique_conditions)}
    means = {}
    for condition in selected_conditions:
        code = condition_to_index[str(condition)]
        rows = order[starts[code] : stops[code]]
        mean = np.asarray(matrix[rows].mean(axis=0)).reshape(-1)
        means[str(condition)] = (mean - control_mean).astype(np.float32, copy=False)
    return means


def _dense_float32(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def initialize_scalable_model(
    model_class,
    adata,
    gene_embeddings: dict[str, np.ndarray],
    *,
    split_name: str = "split",
    latent_dim: int = 30,
    hidden_dim: int = 512,
    training_epochs: int = 200,
    batch_size: int = 500,
    lambda_mi: int = 200,
    sigma_x_sq: int = 1,
    eps: float = 0.001,
    seed: int = 1234,
    model_path: str = "models",
    multi_gene: bool = True,
    control_size: int | None = None,
):
    """Initialize upstream ``Model`` state without its quadratic preprocessing.

    The returned object still uses upstream ``Model.train`` and ``Model.predict`` unchanged.
    This compatibility path intentionally supports the study's validated
    ``use_tg_coord=False`` configuration only.
    """
    import torch
    from sclambda.model import PertDataset
    from torch.utils.data import DataLoader

    model = model_class.__new__(model_class)
    if torch.cuda.is_available():
        model.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        model.device = torch.device("mps")
    else:
        model.device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model.adata = adata
    first_embedding = np.asarray(next(iter(gene_embeddings.values())), dtype=np.float32)
    model.gene_emb = {name: np.asarray(value, dtype=np.float32) for name, value in gene_embeddings.items()}
    model.gene_emb["ctrl"] = np.zeros_like(first_embedding)
    model.x_dim = adata.shape[1]
    model.p_dim = first_embedding.shape[0]
    model.latent_dim = latent_dim
    model.hidden_dim = hidden_dim
    model.training_epochs = training_epochs
    model.batch_size = batch_size
    model.lambda_MI = lambda_mi
    model.sigma_x_sq = sigma_x_sq
    model.eps = eps
    model.model_path = model_path
    model.multi_gene = multi_gene
    model.use_tg_coord = False
    model.ctrl_size = control_size

    conditions = adata.obs["condition"].astype(str).to_numpy()
    unique, condition_indices, embedding_table = build_condition_embedding_table(
        conditions,
        model.gene_emb,
    )
    model.n_tgs = max(len(condition.split("+")) for condition in unique)
    model.pert_emb = {condition: embedding_table[index] for index, condition in enumerate(unique)}
    model.pert_emb_cells = None
    model.pert_emb_cells_tg = {}

    split = adata.obs[split_name].astype(str).to_numpy()
    train_mask = split == "train"
    validation_mask = split == "val"
    control_mask = conditions == "ctrl"
    control_expression = _dense_float32(adata.X[control_mask])
    model.ctrl_mean = control_expression.mean(axis=0, dtype=np.float64).astype(np.float32)
    control_expression -= model.ctrl_mean
    model.ctrl_x = torch.from_numpy(control_expression).float().to(model.device)
    if control_size is not None:
        if control_size > model.ctrl_x.shape[0]:
            raise ValueError(f"control_size={control_size} exceeds {model.ctrl_x.shape[0]} available controls")
        control_indices = np.random.choice(model.ctrl_x.shape[0], control_size, replace=False)
        model.ctrl_idx = torch.from_numpy(control_indices).long().to(model.device)
        model.ctrl_x = model.ctrl_x[model.ctrl_idx]

    print("Materializing centered scLAMBDA training tensors...")
    training_expression = _dense_float32(adata.X[train_mask])
    training_expression -= model.ctrl_mean
    training_embeddings = embedding_table[condition_indices[train_mask]]
    training_expression_tensor = torch.from_numpy(training_expression).float().to(model.device)
    training_embedding_tensor = torch.from_numpy(training_embeddings).float().to(model.device)
    model.train_data = PertDataset(training_expression_tensor, training_embedding_tensor)
    model.train_dataloader = DataLoader(
        model.train_data,
        batch_size=model.batch_size,
        shuffle=True,
    )

    model.adata_train = adata[train_mask]
    model.adata_val = adata[validation_mask]
    model.pert_val = unique[np.isin(unique, conditions[validation_mask])]
    model.pert_delta = selected_centered_condition_means(
        adata.X,
        condition_indices,
        unique,
        model.pert_val,
        model.ctrl_mean,
    )
    return model
