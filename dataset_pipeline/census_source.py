import json
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .config import DatasetSpec
from .runtime import ensure_h5ad_runtime


CELLXGENE_COLLECTIONS_API = "https://api.cellxgene.cziscience.com/curation/v1/collections"


def _require_h5ad_dependencies():
    ensure_h5ad_runtime()


def _require_census_dependencies():
    try:
        import cellxgene_census  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CELLxGENE Census fallback requires cellxgene-census. "
            "Install dataset_pipeline/requirements.txt first."
        ) from exc


def _open_census(census_version: str):
    _require_census_dependencies()
    import cellxgene_census

    return cellxgene_census.open_soma(census_version=census_version)


def _load_json(url: str) -> dict | list:
    with urlopen(url) as response:
        return json.load(response)


def _load_collection_detail(collection_id: str) -> dict:
    return _load_json(f"{CELLXGENE_COLLECTIONS_API}/{collection_id}")


def _normalize_collection_dataset_rows(collections_payload: list[dict]):
    rows = []
    for collection in collections_payload:
        collection_name = collection.get("name")
        collection_id = collection.get("collection_id")
        for dataset in collection.get("datasets", []):
            assets = dataset.get("assets") or []
            h5ad_asset = next((asset for asset in assets if asset.get("filetype") == "H5AD" and asset.get("url")), None)
            rows.append(
                {
                    "dataset_id": dataset.get("dataset_id"),
                    "dataset_title": dataset.get("title"),
                    "collection_name": collection_name,
                    "collection_id": collection_id,
                    "dataset_version_id": dataset.get("dataset_version_id"),
                    "asset_h5ad_url": h5ad_asset.get("url") if h5ad_asset else None,
                    "cell_count": dataset.get("cell_count"),
                    "organism": ", ".join(item.get("label", "") for item in dataset.get("organism", [])),
                    "tissue": ", ".join(item.get("label", "") for item in dataset.get("tissue", [])),
                    "assay": ", ".join(item.get("label", "") for item in dataset.get("assay", [])),
                }
            )
    return rows


def _contains_case_insensitive(value: str | None, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.lower() in (value or "").lower()


def _search_collections_api(
    collection_name_contains: str | None = None,
    dataset_title_contains: str | None = None,
):
    collections_payload = _load_json(CELLXGENE_COLLECTIONS_API)

    if collection_name_contains:
        collections_payload = [
            collection
            for collection in collections_payload
            if collection_name_contains.lower() in (collection.get("name") or "").lower()
        ]

    detailed_collections = [_load_collection_detail(collection["collection_id"]) for collection in collections_payload]
    rows = _normalize_collection_dataset_rows(detailed_collections)
    if dataset_title_contains:
        rows = [
            row
            for row in rows
            if _contains_case_insensitive(row.get("dataset_title"), dataset_title_contains)
        ]
    return rows


def _resolve_dataset_metadata(spec: DatasetSpec) -> dict:
    matches = _search_collections_api(
        collection_name_contains=spec.collection_name_contains,
        dataset_title_contains=spec.dataset_title_contains,
    )

    if spec.dataset_id:
        matches = [match for match in matches if match.get("dataset_id") == spec.dataset_id]

    if not matches:
        raise ValueError(
            f"No CELLxGENE datasets matched collection={spec.collection_name_contains!r} "
            f"title={spec.dataset_title_contains!r} dataset_id={spec.dataset_id!r}"
        )

    return dict(matches[0])


def _download_h5ad_asset(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return output_path


def _subsample_h5ad(path: Path, max_cells: int) -> None:
    _require_h5ad_dependencies()

    import anndata as ad

    adata = ad.read_h5ad(path)
    if adata.n_obs <= max_cells:
        return

    adata = adata[adata.obs.sample(n=max_cells, random_state=0).index].copy()
    adata.write_h5ad(path)


def search_census_datasets(
    census_version: str = "latest",
    collection_name_contains: str | None = None,
    dataset_title_contains: str | None = None,
):
    try:
        with _open_census(census_version) as census:
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()

        if collection_name_contains:
            datasets = datasets[datasets["collection_name"].str.contains(collection_name_contains, case=False, na=False)]
        if dataset_title_contains:
            datasets = datasets[datasets["dataset_title"].str.contains(dataset_title_contains, case=False, na=False)]

        columns = [
            "dataset_id",
            "dataset_title",
            "collection_name",
            "soma_joinid",
        ]
        available = [column for column in columns if column in datasets.columns]
        return datasets[available].to_dict(orient="records")
    except Exception as exc:
        message = str(exc).lower()
        if (
            "s3support" not in message
            and "vfs" not in message
            and "soma" not in message
            and "census fallback" not in message
            and "cellxgene census" not in message
        ):
            raise

    datasets = _search_collections_api(
        collection_name_contains=collection_name_contains,
        dataset_title_contains=dataset_title_contains,
    )
    columns = {
        "dataset_id",
        "dataset_title",
        "collection_name",
        "dataset_version_id",
        "asset_h5ad_url",
    }
    return [{key: value for key, value in row.items() if key in columns} for row in datasets]


def resolve_obs_filter(spec: DatasetSpec) -> str:
    if spec.dataset_id:
        return f"dataset_id == '{spec.dataset_id}'"
    if spec.obs_value_filter:
        return spec.obs_value_filter

    matches = search_census_datasets(
        census_version=spec.census_version,
        collection_name_contains=spec.collection_name_contains,
        dataset_title_contains=spec.dataset_title_contains,
    )
    if not matches:
        raise ValueError(
            f"No CELLxGENE datasets matched collection={spec.collection_name_contains!r} "
            f"title={spec.dataset_title_contains!r}"
        )
    dataset_id = str(matches[0]["dataset_id"])
    return f"dataset_id == '{dataset_id}'"


def download_dataset(spec: DatasetSpec, output_path: Path, max_cells: int | None = None) -> Path:
    try:
        metadata = _resolve_dataset_metadata(spec)
        asset_url = metadata.get("asset_h5ad_url")
        if asset_url:
            path = _download_h5ad_asset(asset_url, output_path)
            if max_cells is not None:
                _subsample_h5ad(path, max_cells)
            return path
    except (URLError, ValueError):
        pass

    _require_census_dependencies()
    _require_h5ad_dependencies()
    import cellxgene_census

    obs_filter = resolve_obs_filter(spec)
    with _open_census(spec.census_version) as census:
        kwargs = {
            "organism": spec.organism,
            "measurement_name": "RNA",
            "obs_value_filter": obs_filter,
        }
        if spec.obs_columns:
            kwargs["column_names"] = {
                "obs": list(spec.obs_columns),
                "var": ["feature_name", "feature_id"],
            }
        adata = cellxgene_census.get_anndata(census, **kwargs)

    if max_cells is not None and adata.n_obs > max_cells:
        adata = adata[adata.obs.sample(n=max_cells, random_state=0).index].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return output_path


def audit_h5ad_metadata(path: Path, required_columns: list[str], state_label_keys: list[str]) -> dict:
    _require_h5ad_dependencies()

    import anndata as ad

    adata = ad.read_h5ad(path)
    obs_columns = list(adata.obs.columns)
    missing_required = [column for column in required_columns if column not in obs_columns]
    present_state_keys = [column for column in state_label_keys if column in obs_columns]
    non_null_counts = {}
    for column in required_columns + present_state_keys:
        if column in adata.obs.columns:
            non_null_counts[column] = int(adata.obs[column].notna().sum())

    return {
        "path": str(path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": obs_columns,
        "missing_required_columns": missing_required,
        "present_state_label_keys": present_state_keys,
        "non_null_counts": non_null_counts,
    }