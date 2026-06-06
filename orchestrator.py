import argparse
import numpy as np

from pathlib import Path
from datasets import DatasetDict
from data_loader import DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, load_data, save_data
from text_cleaning import get_english_stopwords, review_to_words
from bow_features import extract_bow_features


def print_dataset_summary(dataset: DatasetDict) -> None:
    """Print a summary of the dataset, including split names and sizes."""
    print(dataset)
    if "train" in dataset:
        print("train split size:", len(dataset["train"]))
    if "test" in dataset:
        print("test split size:", len(dataset["test"]))


def preprocess_dataset(dataset_name: DatasetDict, output_dir: Path) -> None:
    """Preprocess the dataset by cleaning the review texts and adding a new column with the cleaned text."""
    stop_words = get_english_stopwords(Path(output_dir))
    train_set = dataset_name["train"]
    cleaned_texts = []

    for index in range(len(train_set)):
        raw_review = train_set[index]["text"]
        cleaned_review = review_to_words(raw_review, stop_words)
        cleaned_texts.append(cleaned_review)
    dataset_name["train"] = dataset_name["train"].add_column("cleaned_text", cleaned_texts)

    return dataset_name


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the IMDb dataset download and preprocessing pipeline.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Hugging Face dataset identifier")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save the dataset")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of sample reviews to print")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = load_data(dataset_name=args.dataset_name, output_dir=Path(args.output_dir))
    print("Original dataset summary:")
    print_dataset_summary(dataset)
    dataset_preprocessed = preprocess_dataset(dataset_name = dataset, output_dir=Path(args.output_dir))
    print("Preprocessed dataset summary:")
    print_dataset_summary(dataset_preprocessed)
    save_data(dataset_preprocessed, Path(args.output_dir + "/preprocessed"), Path(args.output_dir))
    # bow_dataset = extract_bow_features(dataset_preprocessed)
    # print_dataset_summary(bow_dataset)
    # print(bow_dataset["train"]["bow_features"][:args.sample_count])
    # print(np.array(bow_dataset["train"]["bow_features"][:args.sample_count]))

if __name__ == "__main__":
    main()

