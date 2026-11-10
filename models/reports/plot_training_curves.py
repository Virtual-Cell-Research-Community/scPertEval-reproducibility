#!/usr/bin/env python3
"""Plot raw training/validation curves for every fold, one column per model.

Reads models/<model>/metrics/<dataset>/fold_<i>.csv directly - no pixi env
needed beyond matplotlib+pandas (the repo's own .venv already has both; none
of the model-specific pixi envs need activating for this).

tabicl is excluded: it has no metrics/ dir at all (no epoch-based training
loop in its logs either - see reports/training_report.py's epoch column, which
comes up "n/a" for every tabicl fold).

Each model logs a genuinely different quantity under "loss"/"val" - conflating
them on one y-axis would be misleading, so every subplot gets its own label
derived from actually reading each model's training code (not the CSV header
alone):
  - gears:    train_mse/val_mse       - plain per-gene MSE (gears' own evaluate())
  - presage:  train_loss/val_loss     - plain MSE (model_harness.py: F.mse_loss(pred, expr))
  - scgpt:    train_loss/val_pearson  - masked MSE (MLM-style, masked genes only) for
              train; val is NOT a loss, it's Pearson r of predicted vs true response
  - sclambda: train_loss/val_corr     - train is a composite VAE objective (reconstruction
              + adversarial MINE mutual-info term), not plain MSE, hence the much larger
              raw scale (~500 vs ~0.01 for MSE-based models); val is Pearson r between
              predicted and true perturbation-induced expression delta
  - state:    train_loss/val_loss     - energy-distance distributional loss (geomloss
              SamplesLoss, config.yaml's distributional_loss: energy), comparing predicted
              vs real perturbed-cell-population distributions - NOT a pointwise MSE
              (state also logs a separate decoder_loss, a genuine MSE, not plotted here)

presage's and state's metrics.csv come from PyTorch Lightning's CSVLogger, which
interleaves train-step rows and val-epoch rows with blank cells for whatever
wasn't logged that row - handled here by dropping rows where the target column
is empty, independently per column.

Each validation subplot also marks the epoch/step of the checkpoint actually
saved to disk (a red dot) - the exact selection criterion was read out of each
model's own code, not assumed to be "best point on the plotted curve":
  - gears:    argmin(val_mse_de) - the DE-gene-subset MSE, a DIFFERENT column
              from the plotted val_mse (gears/gears.py's train(): `if
              val_metrics['mse_de'] < min_val: best_model = deepcopy(...)`) -
              so the red dot's x-position can land somewhere the plotted
              val_mse curve doesn't look locally best, because it wasn't
              selected by that curve.
  - presage:  argmin(val_loss) - Lightning ModelCheckpoint(monitor="val_loss",
              mode="min"), matches the plotted curve directly.
  - scgpt:    argmax(val_pearson) - its own patience loop's `if val_score >
              best_val_corr: best_model = ...`, matches the plotted curve.
  - sclambda: argmax(val_corr) - `if corr_val > corr_val_best: model_best =
              ...`, matches the plotted curve.
  - state:    argmin(val_loss) - Lightning ModelCheckpoint(monitor="val_loss",
              mode="min"), matches the plotted curve directly.

Usage:
    python3 reports/plot_training_curves.py
    python3 reports/plot_training_curves.py --dataset replogle22k562 --out figures/training_curves.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent  # models/ - one level up from reports/
N_FOLDS = 5

# Categorical palette, fixed order (dataviz skill's validated default, slots 1-5) -
# one color per fold, held constant across every subplot.
FOLD_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]

MODEL_SPECS = {
    "gears": dict(
        x="epoch", train_col="train_mse", val_col="val_mse",
        train_label="Train MSE (test genes)", val_label="Val MSE (test genes)",
        best_col="val_mse_de", best_direction="min",
    ),
    "presage": dict(
        x="step", train_col="train_loss", val_col="val_loss",
        train_label="Train MSE", val_label="Val MSE",
        best_col="val_loss", best_direction="min",
    ),
    "scgpt": dict(
        x="epoch", train_col="train_loss", val_col="val_pearson",
        train_label="Train masked MSE (MLM)", val_label="Val Pearson r (response)",
        best_col="val_pearson", best_direction="max",
    ),
    "sclambda": dict(
        x="epoch", train_col="train_loss", val_col="val_corr",
        train_label="Train loss (VAE recon + adversarial MI, a.u.)", val_label="Val Pearson r (Δexpr)",
        best_col="val_corr", best_direction="max",
    ),
    "state": dict(
        x="step", train_col="train_loss", val_col="val_loss",
        train_label="Train energy-distance loss", val_label="Val energy-distance loss",
        best_col="val_loss", best_direction="min",
    ),
}
BEST_MARKER_COLOR = "#e34948"  # dataviz palette's red slot (8) - reserved for this, not a fold
MODEL_TITLES = {"gears": "GEARS", "presage": "PRESAGE", "scgpt": "scGPT", "sclambda": "scLAMBDA", "state": "STATE"}
MODELS_ORDER = ["gears", "presage", "scgpt", "sclambda", "state"]


def load_fold_series(model, fold, dataset, col, x_col):
    path = MODELS_DIR / model / "metrics" / dataset / f"fold_{fold}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if col not in df.columns:
        return None
    df = df[[x_col, col]].dropna(subset=[col])
    if df.empty:
        return None
    return df[x_col].to_numpy(), df[col].to_numpy()


def find_best_point(model, fold, dataset, spec):
    """(x, plotted-val-col-value) of the row that was actually selected as the saved
    checkpoint, per that model's own criterion - see module docstring. best_col and
    val_col can differ (gears), so this reads both off the SAME row rather than
    matching by x, in case the two columns are ever logged on different rows."""
    path = MODELS_DIR / model / "metrics" / dataset / f"fold_{fold}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    best_col, val_col, x_col = spec["best_col"], spec["val_col"], spec["x"]
    if best_col not in df.columns or val_col not in df.columns:
        return None
    sub = df.dropna(subset=[best_col, val_col])
    if sub.empty:
        return None
    idx = sub[best_col].idxmin() if spec["best_direction"] == "min" else sub[best_col].idxmax()
    row = df.loc[idx]
    return row[x_col], row[val_col]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="replogle22k562")
    ap.add_argument("--models", nargs="+", default=MODELS_ORDER, choices=MODELS_ORDER)
    ap.add_argument("--out", default="figures/training_curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(
        2, len(args.models), figsize=(4 * len(args.models), 7),
        squeeze=False, facecolor="white",
    )

    for col_idx, model in enumerate(args.models):
        spec = MODEL_SPECS[model]
        ax_train, ax_val = axes[0, col_idx], axes[1, col_idx]
        for fold in range(N_FOLDS):
            color = FOLD_COLORS[fold]
            train_xy = load_fold_series(model, fold, args.dataset, spec["train_col"], spec["x"])
            if train_xy:
                ax_train.plot(*train_xy, color=color, linewidth=2, label=f"fold {fold}")
            val_xy = load_fold_series(model, fold, args.dataset, spec["val_col"], spec["x"])
            if val_xy:
                ax_val.plot(*val_xy, color=color, linewidth=2, label=f"fold {fold}")

            best_xy = find_best_point(model, fold, args.dataset, spec)
            if best_xy:
                ax_val.scatter(*best_xy, color=BEST_MARKER_COLOR, s=45, zorder=5,
                                edgecolors="white", linewidths=0.8, label="saved checkpoint")

        ax_train.set_title(MODEL_TITLES[model], fontsize=17, fontweight="bold")
        ax_train.set_ylabel(spec["train_label"], fontsize=10)
        ax_val.set_ylabel(spec["val_label"], fontsize=10)
        ax_val.set_xlabel(spec["x"], fontsize=10)
        for ax in (ax_train, ax_val):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, linewidth=0.5, alpha=0.3)
            ax.tick_params(labelsize=9)

    # Row banners.
    fig.text(0.005, 0.72, "Training", rotation=90, va="center", fontsize=12, fontweight="bold")
    fig.text(0.005, 0.27, "Validation", rotation=90, va="center", fontsize=12, fontweight="bold")

    # Fold-color legend from the training row; the "saved checkpoint" marker's legend entry
    # gets added once per fold per column (same label) - dedupe down to a single swatch.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes[1, :]:
        val_handles, val_labels = ax.get_legend_handles_labels()
        if "saved checkpoint" in val_labels:
            handles.append(val_handles[val_labels.index("saved checkpoint")])
            labels.append("saved checkpoint")
            break
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Raw training/validation curves — {args.dataset}", fontsize=20, fontweight="bold", y=1.04)
    fig.tight_layout(rect=[0.02, 0.03, 1, 1])

    out_path = MODELS_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
