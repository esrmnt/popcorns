from pathlib import Path
from typing import Optional
from datasets import DatasetDict, load_dataset

DEFAULT_DATASET_NAME = "stanfordnlp/imdb"
DEFAULT_OUTPUT_DIR = Path("dataset")


def load_data(dataset_name: str = DEFAULT_DATASET_NAME) -> DatasetDict:
    """Load the IMDb dataset from the Hugging Face hub."""
    dataset = load_dataset(dataset_name)
    return dataset


def save_data(dataset: DatasetDict, target_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Save a Hugging Face dataset to disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {target_dir}")
    dataset.save_to_disk(str(target_dir))
