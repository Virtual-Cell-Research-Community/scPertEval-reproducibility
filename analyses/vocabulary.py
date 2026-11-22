"""Reader-facing names shared by all manuscript figures.

One label per evaluation choice, anchored on the section titles of the
manuscript's taxonomy (Fig. 1). The Atlas census and the ranking-sensitivity
analysis use different internal codes for the same choices; both are mapped
here so that a choice carries the same name in every panel and in the text.
"""

from __future__ import annotations

AXIS_LABELS = {
    "representation": "Representation",
    "embedding": "Embedding",
    "effect_isolation": "Perturbation effect isolation",
    "feature_selection": "Feature selection",
    "target_gene_policy": "Target-gene policy",
    "feature_weighting": "Feature weighting",
    "metric": "Metric",
    "aggregation": "Aggregation",
}
"""Evaluation choice → label. Order follows the taxonomy (representation → metric → reporting)."""

ATLAS_AXES = (
    "representation",
    "embedding",
    "effect_isolation",
    "feature_selection",
    "target_gene_policy",
    "feature_weighting",
    "metric",
)
"""Evaluation choices in the order of the ``__``-separated Atlas matrix column."""

RANKING_FACTORS = {
    "evaluation_family": "representation",
    "expression_space": "embedding",
    "centering": "effect_isolation",
    "feature_panel": "feature_selection",
    "target_policy": "target_gene_policy",
    "weighting": "feature_weighting",
    "score_function": "metric",
    "aggregation": "aggregation",
}
"""Ranking-sensitivity factor code → evaluation choice."""

VALUE_LABELS = {
    "representation": {
        "SC": "Single-cell",
        "PBULK": "Pseudobulk",
        "DE": "DE statistics",
        "COMP": "Composition",
        "MASS": "Abundance",
        "VAR": "Variance",
    },
    "embedding": {
        "GEX": "Genes",
        "PCA": "PCA",
        "FM": "Foundation model",
        "BIO": "Prior knowledge",
    },
    "effect_isolation": {
        "uncentered": "None",
        "control-centered": "Control",
        "pert-centered": "Other perturbations",
    },
}

TARGET_CONTEXT_LABELS = {
    "controls_only": "Zero-shot",
    "perturbation_responses_available": "Few-shot",
}

BENCHMARK_TASKS = {
    "essential_perturbation_generalization": "Perturbation generalization",
    "genetic_context_transfer": "Genetic context transfer",
    "chemical_context_transfer": "Chemical context transfer",
}
"""Benchmark task id → label, in display order."""

CENSUS_TASKS = {
    "essential_perturbation_generalization": "Genetic single, Few-shot",
    "genetic_context_transfer": "Genetic single, Zero-shot",
    "chemical_context_transfer": "Chemical single, Zero-shot",
}
"""Benchmark task → the census task (modality, order, target context) it is an instance of."""


def axis_label(choice: str) -> str:
    return AXIS_LABELS[RANKING_FACTORS.get(choice, choice)]
