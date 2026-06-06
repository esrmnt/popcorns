from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

DEFAULT_DATASET_NAME = "stanfordnlp/imdb"
DEFAULT_OUTPUT_DIR = Path("dataset")
DEFAULT_LOCAL_DATASET_PATH = Path("dataset/raw")


def load_data(dataset_name: str = DEFAULT_DATASET_NAME, path: Path = DEFAULT_LOCAL_DATASET_PATH) -> DatasetDict:
    """Load a dataset from a local path when available, otherwise download it from the Hugging Face hub."""
    local_path = Path(path)

    if local_path.exists():
        print(f"Loading dataset '{dataset_name}' from local path '{local_path}'...")
        return load_from_disk(str(local_path))

    print(f"Loading dataset '{dataset_name}' from Hugging Face hub...")
    dataset = load_dataset(dataset_name)
    save_data(dataset, local_path)
    return dataset


def save_data(dataset: DatasetDict, target_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Save a Hugging Face dataset to disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {target_dir}")
    dataset.save_to_disk(str(target_dir))
