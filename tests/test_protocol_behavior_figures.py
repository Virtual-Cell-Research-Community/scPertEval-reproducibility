from __future__ import annotations

from analyses.manuscript_figures.protocol_behavior import PROTOCOLS, criterion_latex, protocol_configuration


def test_configuration_follows_scperteval_protocols_and_default_controls():
    config = protocol_configuration().set_index("protocol")

    assert list(config.index) == PROTOCOLS
    assert config.loc["pearson_pert", ["centering", "negative"]].tolist() == ["Pert.", "Ctrl mean"]
    assert config.loc["pearson", ["positive", "negative"]].tolist() == ["Interp. TD", "Pert. mean"]
    assert config.loc["rank", "negative"] == "Global mean"
    assert config.loc["de_auroc", ["representation", "positive", "negative"]].tolist() == ["DE", "TD", "Pert. cells"]
    assert config.loc["energy_distance_pca_k=50", "space"] == "PCA-50"
    assert config.loc["wmse_exp4", "weight"] == "4"


def test_criterion_latex_labels():
    assert criterion_latex("Welch t-test |stat|>2") == "$t$-test, $|t| > 2$"
    assert criterion_latex("MWU |delta|>0.15") == r"MWU, $|\delta| > 0.15$"
    assert criterion_latex("MWU padj<0.05") == r"MWU, $p_{\mathrm{adj}} < 0.05$"
