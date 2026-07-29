import argparse
import numpy as np

from pathlib import Path
from datasets import DatasetDict
from data_loader import DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, load_data, save_data
from text_cleaning import get_english_stopwords, text_to_words, text_to_words
from bow_features import extract_bow_features, train_word2vec_model


def print_dataset_summary(dataset: DatasetDict) -> None:
    """Print a summary of the dataset, including split names and sizes."""
    print(dataset)
    if "train" in dataset:
        print("train split size:", len(dataset["train"]))
    if "test" in dataset:
        print("test split size:", len(dataset["test"]))


def preprocess_dataset(dataset: DatasetDict, dataset_name:str, output_dir: Path) -> None:
    """Preprocess the dataset by cleaning the texts and adding a new column with the cleaned text."""
    path = Path(output_dir) / dataset_name.replace("/", "_") / "preprocessed"

    if path.exists():
        print(f"Loading preprocessed dataset from {path}...")
        return load_data(dataset_name=dataset_name, path=path, output_dir=output_dir)

    print(f"Preprocessing texts...")
    stop_words = get_english_stopwords(Path(output_dir))
    train_set = dataset["train"]
    cleaned_texts = []

    for index in range(len(train_set)):
        raw_text = train_set[index]["text"]
        cleaned_text = text_to_words(raw_text, stop_words)
        cleaned_texts.append(cleaned_text)
        print(f"Processed {index + 1}/{len(train_set)} texts", end="\r")
    dataset["train"] = dataset["train"].add_column("cleaned_text", cleaned_texts)

    save_data(dataset, path)

    return dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate the IMDb dataset download and preprocessing pipeline.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME, help="Hugging Face dataset identifier")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save the dataset")
    parser.add_argument("--sample-count", type=int, default=1, help="Number of samples to print on console")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    dataset = load_data(dataset_name=args.dataset_name, output_dir=Path(args.output_dir))
    # print("Original dataset summary:")
    # print_dataset_summary(dataset)

    dataset_preprocessed = preprocess_dataset(dataset = dataset, dataset_name=args.dataset_name, output_dir=Path(args.output_dir))
    # print("Preprocessed dataset summary:")
    # print_dataset_summary(dataset_preprocessed)

    # print(f"\nSample cleaned texts (first {args.sample_count}):")
    # for i in range(args.sample_count):
    #     print("-----------------------------")
    #     print(f"Original Text {i + 1}: {dataset_preprocessed['train'][i]['text']}")
    #     print("\n")
    #     print(f"Cleaned Text {i + 1}: {dataset_preprocessed['train'][i]['cleaned_text']}")

    w2v_model = train_word2vec_model(dataset_preprocessed, text_column="cleaned_text")
    print("Vocabulary size:", len(w2v_model.wv))
    print("Vector for 'good':", w2v_model.wv["good"][:5])

    # print("Similarity between 'good' and 'great':", w2v_model.wv.similarity("good", "great"))

if __name__ == "__main__":
    main()

