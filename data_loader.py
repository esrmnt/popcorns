from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset

DEFAULT_DATASET_NAME = "stanfordnlp/imdb"
DEFAULT_OUTPUT_DIR = Path("dataset")


def load_imdb_dataset(dataset_name: str = DEFAULT_DATASET_NAME) -> DatasetDict:
    """Load the IMDb dataset from the Hugging Face hub."""
    dataset = load_dataset(dataset_name)
    save_dataset(dataset, DEFAULT_OUTPUT_DIR)
    return dataset


def save_dataset(dataset: DatasetDict, target_dir: Optional[Path] = None) -> None:
    """Save a Hugging Face dataset to disk."""
    if target_dir is None:
        target_dir = DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target_dir))
