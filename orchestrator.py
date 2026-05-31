import argparse

from pathlib import Path
from datasets import DatasetDict
from data_loader import DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, load_imdb_dataset, save_dataset
from text_cleaning import get_english_stopwords, review_to_words


def print_dataset_summary(dataset: DatasetDict) -> None:
    print(dataset)
    if "train" in dataset:
        print("train split size:", len(dataset["train"]))
    if "test" in dataset:
        print("test split size:", len(dataset["test"]))


def print_sample_reviews(dataset: DatasetDict, sample_count: int = 3) -> None:
    stop_words = get_english_stopwords()
    train_set = dataset["train"]

    print("\nSample reviews from the train split:\n")
    for index in range(min(sample_count, len(train_set))):
        raw_review = train_set[index]["text"]
        label = train_set[index]["label"]
        cleaned_review = review_to_words(raw_review, stop_words)

        print(f"--- Review {index} (label={label}) ---")
        print(raw_review[:500].replace("\n", " "))
        print("\nCleaned:")
        print(cleaned_review[:500])
        print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the IMDb dataset download and preprocessing pipeline.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Hugging Face dataset identifier")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save the dataset")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of sample reviews to print")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = load_imdb_dataset(args.dataset_name)
    print_dataset_summary(dataset)
    print_sample_reviews(dataset, sample_count=args.sample_count)


if __name__ == "__main__":
    main()

