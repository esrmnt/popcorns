import argparse

from pathlib import Path
from datasets import DatasetDict
from data_loader import DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, load_data, save_data
from text_cleaning import get_english_stopwords, review_to_words


def print_dataset_summary(dataset: DatasetDict) -> None:
    print(dataset)
    if "train" in dataset:
        print("train split size:", len(dataset["train"]))
    if "test" in dataset:
        print("test split size:", len(dataset["test"]))


def preprocess_dataset(dataset: DatasetDict, output_dir: Path) -> None:
    stop_words = get_english_stopwords()
    train_set = dataset["train"]
    cleaned_texts = []

    for index in range(len(train_set)):
        raw_review = train_set[index]["text"]
        cleaned_review = review_to_words(raw_review, stop_words)
        cleaned_texts.append(cleaned_review)
    dataset["train"] = dataset["train"].add_column("cleaned_text", cleaned_texts)

    print(dataset)
    print(dataset["train"])
    return dataset
    

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the IMDb dataset download and preprocessing pipeline.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Hugging Face dataset identifier")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save the dataset")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of sample reviews to print")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = load_data(args.dataset_name)
    save_data(dataset, Path(args.output_dir + "/raw"))
    dataset_preprocessed = preprocess_dataset(dataset, Path(args.output_dir + "/preprocessed"))
    save_data(dataset_preprocessed, Path(args.output_dir + "/preprocessed"))

if __name__ == "__main__":
    main()

