"""Dataset-specific raw-count preparation adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData, read_h5ad
from rdkit import Chem
from rdkit.Chem import SaltRemover, inchi

from workflows.preparation_utils import (
    add_crispri_residual_expression,
    add_vst_statistics,
    assert_raw_counts,
    column_variance,
    read_cell_cycle_genes,
)

SCIPLEX_EXPRESSION = "SrivatsanTrapnell2020_sciplex3.h5ad"
SCIPLEX_METADATA = "Supplementary_Table_3.txt"
SCIPLEX_BRD_MAP = "l1000_sciChem.tsv"
SCIPLEX_NAME_CORRECTIONS = {
    "Glesatinib?(MGCD265)": "Glesatinib(MGCD265)",
    "SNS-314": "SNS-314 Mesylate",
}


def merge_disambiguated_gene_names(source: AnnData) -> AnnData:
    """Merge scPerturb columns that encode duplicate symbols with Ensembl suffixes."""
    names = pd.Index(source.var_names)
    bases = names.str.replace(r"_ENSG\d+$", "", regex=True)
    merged_names = np.where(bases.duplicated(keep=False), bases, names)
    if not pd.Index(merged_names).duplicated().any():
        return source

    codes, unique_names = pd.factorize(merged_names, sort=False)
    gather = sp.csr_matrix(
        (np.ones(source.n_vars, dtype=np.int8), (np.arange(source.n_vars), codes)),
        shape=(source.n_vars, len(unique_names)),
    )
    merged_x = (source.X @ gather).tocsr()
    var = source.var.copy()
    var.index = pd.Index(merged_names)
    var = var.groupby(level=0, sort=False).first().reindex(unique_names)
    var["n_merged"] = np.bincount(codes, minlength=len(unique_names))
    return AnnData(X=merged_x, obs=source.obs.copy(), var=var)


def harmonize_scperturb(source: AnnData, *, merge_disambiguated: bool = False) -> AnnData:
    """Reduce a scPerturb object to raw counts and compact perturbation fields."""
    required_obs = {"batch", "gene", "gene_id", "guide_id"}
    missing = sorted(required_obs - set(source.obs.columns))
    if missing:
        raise ValueError(f"scPerturb source is missing obs fields: {', '.join(missing)}")

    assert_raw_counts(source.X)
    source.X = sp.csr_matrix(source.X, dtype=np.float32)
    source.uns.pop("log1p", None)
    source.layers.clear()
    if merge_disambiguated:
        source = merge_disambiguated_gene_names(source)
    source.obs = source.obs[["batch", "gene", "gene_id", "guide_id"]].rename(
        columns={"gene": "target_gene", "gene_id": "target_gene_ensembl"}
    )
    for column in ["batch", "target_gene", "target_gene_ensembl", "guide_id"]:
        source.obs[column] = source.obs[column].astype("category")
    source.obs["perturbation"] = source.obs["target_gene"]
    source.obs["is_control"] = source.obs["target_gene"] == "non-targeting"
    source.obs["guide_target"] = source.obs["target_gene"]
    source.obs["n_guides"] = np.int8(1)
    if not source.obs["is_control"].any():
        raise ValueError("scPerturb source has no non-targeting control cells")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("scPerturb source has duplicate cell or gene identifiers")
    return source


def prepare_scperturb(
    source_path: Path,
    *,
    cell_cycle_path: Path,
    merge_disambiguated: bool = False,
    filter_residual_expression: bool = True,
    compute_vst: bool = True,
) -> AnnData:
    """Apply the shared CRISPRi QC workflow while preserving raw counts in X."""
    adata = harmonize_scperturb(read_h5ad(source_path), merge_disambiguated=merge_disambiguated)
    adata.obs["control_class"] = pd.Categorical(
        np.where(adata.obs["is_control"], "non-targeting", "targeted")
    )

    adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["total_counts_log10"] = np.log10(adata.obs["total_counts"])
    adata.obs["detected_genes"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(r"^HB[ABDEGZ]")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "hb", "ribo"],
        inplace=True,
        log1p=False,
        percent_top=None,
    )
    adata = adata[
        (adata.obs["total_counts"] >= 2_500) & (adata.obs["pct_counts_mt"] < 25),
        adata.var["total_counts"] >= 200,
    ].copy()

    raw_counts = adata.X.copy()
    sc.pp.normalize_total(adata)
    add_crispri_residual_expression(adata)
    pre_residual_names = adata.obs_names.copy()
    if filter_residual_expression:
        keep = ((adata.obs["residual_expression"] < 0.5) | adata.obs["residual_expression"].isna()).to_numpy()
    else:
        keep = np.ones(adata.n_obs, dtype=bool)
    adata = adata[keep].copy()
    raw_counts = raw_counts[pre_residual_names.get_indexer(adata.obs_names)]

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    s_genes, g2m_genes = read_cell_cycle_genes(cell_cycle_path)
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    pre_target_names = adata.obs_names.copy()
    counts = adata.obs["target_gene"].value_counts()
    keep_targets = counts[counts >= 10].index
    adata = adata[adata.obs["target_gene"].isin(keep_targets)].copy()
    raw_counts = raw_counts[pre_target_names.get_indexer(adata.obs_names)]

    adata.X = raw_counts.copy()
    sc.pp.normalize_total(adata)
    adata.var["mean"] = adata.X.mean(axis=0).A1
    adata.var["var"] = column_variance(adata.X)
    positive_mean = adata.var.loc[adata.var["mean"] > 0, "mean"].min()
    positive_var = adata.var.loc[adata.var["var"] > 0, "var"].min()
    adata.var["mean"] = np.where(adata.var["mean"] <= 0, positive_mean, adata.var["mean"])
    adata.var["var"] = np.where(adata.var["var"] <= 0, positive_var, adata.var["var"])
    adata.var["log10_mean"] = np.log10(adata.var["mean"])
    adata.var["log10_var"] = np.log10(adata.var["var"])
    adata.var["disp"] = adata.var["var"] / adata.var["mean"]
    adata.var["disp_rank"] = adata.var["disp"].rank(ascending=False)
    adata.var["log10_disp"] = np.log10(adata.var["disp"])
    adata.var["mean_log1p_norm_count"] = np.ravel(np.mean(adata.X, axis=0))
    if compute_vst:
        add_vst_statistics(adata)

    adata.X = raw_counts.copy()
    adata.uns.pop("log1p", None)
    assert_raw_counts(adata.X)
    return adata


def canonicalize_parent_smiles(value: object) -> str | None:
    """Return a canonical desalted SMILES without discarding the source string."""
    if pd.isna(value) or not str(value).strip():
        return None
    molecule = Chem.MolFromSmiles(str(value).strip().split()[0])
    if molecule is None:
        return None
    fragments = [
        fragment
        for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
        if all(atom.GetAtomicNum() != 0 for atom in fragment.GetAtoms())
    ]
    if fragments:
        molecule = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
    parent = SaltRemover.SaltRemover().StripMol(molecule, dontRemoveEverything=True)
    return Chem.MolToSmiles(parent, canonical=True)


def smiles_to_inchikey(value: object) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    molecule = Chem.MolFromSmiles(str(value))
    return inchi.MolToInchiKey(molecule) if molecule is not None else None


def load_compound_metadata(metadata_path: Path, brd_map_path: Path) -> pd.DataFrame:
    """Load published compounds and derive deterministic identifiers."""
    metadata = pd.read_csv(metadata_path, sep="\t")
    required = ["catalog_number", "name", "SMILES", "CAS.Number", "M.w."]
    missing = sorted(set(required) - set(metadata.columns))
    if missing:
        raise ValueError(f"compound metadata is missing fields: {', '.join(missing)}")
    metadata = metadata[required].copy()
    for column in ["catalog_number", "name", "SMILES"]:
        metadata[column] = metadata[column].astype("string").str.strip()
    metadata = metadata.dropna(subset=["name"]).drop_duplicates()
    conflicts = metadata.groupby("name", observed=True).size()
    if (conflicts > 1).any():
        raise ValueError("compound metadata has conflicting rows")

    brd_map = pd.read_csv(
        brd_map_path,
        sep="\t",
        header=None,
        names=["catalog_number", "brd_id"],
        dtype="string",
        on_bad_lines="skip",
    ).dropna()
    brd_map = brd_map.apply(lambda column: column.str.strip())
    brd_map = brd_map.groupby("catalog_number", as_index=False, observed=True)["brd_id"].agg(
        lambda values: "|".join(sorted(set(values)))
    )
    metadata = metadata.merge(brd_map, on="catalog_number", how="left", validate="one_to_one")
    metadata = metadata.rename(
        columns={
            "name": "perturbation",
            "SMILES": "source_smiles",
            "CAS.Number": "cas_number",
            "M.w.": "molecular_weight",
        }
    )
    metadata["canonical_smiles"] = metadata["source_smiles"].map(canonicalize_parent_smiles)
    metadata["inchikey"] = metadata["canonical_smiles"].map(smiles_to_inchikey)
    metadata["inchikey_connectivity"] = metadata["inchikey"].str.slice(0, 14)
    if metadata["inchikey"].isna().any():
        raise ValueError("could not derive all compound InChIKeys")
    return metadata.set_index("perturbation")


def prepare_sciplex(source_dir: Path, *, expected_compounds: int = 188) -> AnnData:
    """Prepare sci-Plex3 while preserving raw counts and chemistry metadata."""
    compound_metadata = load_compound_metadata(source_dir / SCIPLEX_METADATA, source_dir / SCIPLEX_BRD_MAP)
    source = read_h5ad(source_dir / SCIPLEX_EXPRESSION)
    required_obs = {
        "ncounts",
        "well",
        "plate",
        "cell_line",
        "replicate",
        "time",
        "dose_value",
        "dose_unit",
        "perturbation",
        "target",
        "pathway",
        "pathway_level_1",
        "pathway_level_2",
    }
    missing = sorted(required_obs - set(source.obs.columns))
    if missing:
        raise ValueError(f"sci-Plex source is missing obs fields: {', '.join(missing)}")
    assert_raw_counts(source.X)
    assigned = source.obs["perturbation"].notna() & source.obs["cell_line"].notna() & source.obs["dose_value"].notna()
    result = source[assigned].copy()
    result.X = sp.csr_matrix(result.X, dtype=np.float32)
    result.layers.clear()
    result.obsm.clear()
    result.varm.clear()
    result.uns.clear()

    canonical_names = result.obs["perturbation"].astype("string").str.strip().replace(SCIPLEX_NAME_CORRECTIONS)
    is_control = canonical_names.str.casefold() == "control"
    missing_compounds = pd.Index(canonical_names[~is_control].unique()).difference(compound_metadata.index)
    if len(missing_compounds):
        raise ValueError(f"missing compound metadata for: {', '.join(missing_compounds)}")
    chemistry = compound_metadata.reindex(canonical_names).reset_index(drop=True)
    chemistry.index = result.obs_names
    chemistry.loc[is_control.to_numpy(), :] = pd.NA

    obs = pd.DataFrame(index=result.obs_names)
    obs["perturbation"] = canonical_names.to_numpy()
    obs["is_control"] = is_control.to_numpy(dtype=bool)
    obs["cell_line"] = result.obs["cell_line"].to_numpy()
    obs["dose"] = result.obs["dose_value"].to_numpy(dtype=np.float32)
    obs["dose_unit"] = result.obs["dose_unit"].to_numpy()
    obs["batch"] = result.obs["plate"].to_numpy()
    for column in ["plate", "well", "replicate", "time"]:
        obs[column] = result.obs[column].to_numpy()
    obs["total_counts"] = result.obs["ncounts"].to_numpy()
    obs["compound_target"] = result.obs["target"].to_numpy()
    for column in ["pathway", "pathway_level_1", "pathway_level_2"]:
        obs[column] = result.obs[column].to_numpy()
    for column in chemistry:
        obs[column] = chemistry[column].to_numpy()
    categorical = [column for column in obs if column not in {"is_control", "dose", "time", "total_counts", "molecular_weight"}]
    for column in categorical:
        obs[column] = obs[column].astype("category")
    result.obs = obs

    treated = ~result.obs["is_control"]
    if not result.obs["is_control"].any():
        raise ValueError("sci-Plex source has no vehicle controls")
    if result.obs.loc[treated, "inchikey"].isna().any():
        raise ValueError("treated sci-Plex cells are missing InChIKeys")
    if result.obs.loc[treated, "perturbation"].nunique() != expected_compounds:
        raise ValueError(f"expected exactly {expected_compounds} sci-Plex3 compounds")
    assert_raw_counts(result.X)
    return result
