from .config import DatasetSpec, PipelineConfig


def enabled_datasets(config: PipelineConfig) -> list[DatasetSpec]:
    return [dataset for dataset in config.datasets if dataset.enabled]


def get_dataset_spec(config: PipelineConfig, name: str) -> DatasetSpec:
    for dataset in config.datasets:
        if dataset.name == name:
            return dataset
    raise KeyError(f"Unknown dataset spec: {name}")