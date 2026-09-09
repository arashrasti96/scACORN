import re
from pathlib import Path

from .config import DatasetSpec, PipelineConfig
from .runtime import ensure_h5ad_runtime
from .schemas import NormalizedCellRecord


def _require_anndata_dependencies():
    ensure_h5ad_runtime()


def _dense_row(matrix_row):
    if hasattr(matrix_row, "toarray"):
        return matrix_row.toarray().ravel()
    return matrix_row.ravel()


ENSEMBL_GENE_ID_RE = re.compile(r"^ENSG[0-9]+(?:\.[0-9]+)?$")


def _clean_gene_name(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _normalize_ensembl_gene_id(value: str | None) -> str | None:
    if not value:
        return None
    match = ENSEMBL_GENE_ID_RE.match(value)
    if not match:
        return None
    return value.split(".", 1)[0]


def _map_ensembl_ids_to_symbols(values: list[str | None]) -> dict[str, str]:
    ids = sorted({gene_id for value in values if (gene_id := _normalize_ensembl_gene_id(value))})
    if not ids:
        return {}

    try:
        import mygene
    except Exception:
        return {}

    client = mygene.MyGeneInfo()
    try:
        response = client.querymany(
            ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=False,
            returnall=False,
            verbose=False,
        )
    except Exception:
        return {}

    mapping = {}
    for row in response:
        query = _normalize_ensembl_gene_id(str(row.get("query", "")).strip())
        symbol = _clean_gene_name(row.get("symbol"))
        if query and symbol and not ENSEMBL_GENE_ID_RE.match(symbol):
            mapping[query] = symbol
    return mapping


def _resolve_gene_names(adata) -> list[str | None]:
    candidate_columns = [
        "feature_name",
        "gene_name",
        "gene_symbols",
        "symbol",
        "features",
    ]

    fallback_names = [_clean_gene_name(value) for value in adata.var_names.tolist()]
    fallback_mapping = _map_ensembl_ids_to_symbols(fallback_names)

    for column in candidate_columns:
        if column not in adata.var.columns:
            continue
        values = adata.var[column].astype(str).tolist()
        resolved = []
        usable = 0
        for fallback, value in zip(adata.var_names.tolist(), values):
            text = _clean_gene_name(value)
            if text and not ENSEMBL_GENE_ID_RE.match(text):
                resolved.append(text)
                usable += 1
            else:
                fallback_text = _clean_gene_name(fallback)
                fallback_id = _normalize_ensembl_gene_id(fallback_text)
                mapped = fallback_mapping.get(fallback_id) if fallback_id else fallback_text
                resolved.append(mapped if mapped and not ENSEMBL_GENE_ID_RE.match(mapped) else None)
        if usable:
            return resolved

    return [
        fallback_mapping.get(_normalize_ensembl_gene_id(name), name)
        if _normalize_ensembl_gene_id(name)
        else name
        for name in fallback_names
    ]


def _rank_genes_for_cell(adata, row_index: int, top_genes: int, gene_names: list[str | None]) -> list[str]:
    import numpy as np

    row = _dense_row(adata.X[row_index])
    if row.size == 0:
        return []

    top_indices = np.argsort(row)[::-1]
    genes = []
    seen = set()
    for index in top_indices:
        gene = gene_names[index]
        if gene and not ENSEMBL_GENE_ID_RE.match(gene) and gene not in seen:
            genes.append(gene)
            seen.add(gene)
        if len(genes) >= top_genes:
            break
    return genes


def normalize_h5ad(path: Path, spec: DatasetSpec, config: PipelineConfig) -> list[NormalizedCellRecord]:
    _require_anndata_dependencies()

    import anndata as ad

    adata = ad.read_h5ad(path)
    dataset_id = spec.dataset_id or spec.name
    records = []
    gene_names = _resolve_gene_names(adata)

    for row_index in range(adata.n_obs):
        genes = _rank_genes_for_cell(adata, row_index, config.defaults.top_genes, gene_names)
        if not genes:
            continue

        row = adata.obs.iloc[row_index]
        sample_id = str(row.get("cell_id") or row.get("obs_id") or row.name)
        metadata = {}
        for key, value in row.items():
            if value is None:
                continue
            text = str(value)
            if text and text.lower() != "nan":
                metadata[key] = text
        metadata.setdefault("organism", spec.organism)

        records.append(
            NormalizedCellRecord(
                sample_id=f"{spec.name}:{sample_id}",
                source_name=spec.name,
                source_dataset_id=dataset_id,
                organism=spec.organism,
                genes=genes,
                metadata=metadata,
            )
        )

    return records