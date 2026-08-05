import argparse
import numpy as np

from pathlib import Path
from datasets import DatasetDict
from data_loader import DEFAULT_DATASET_NAME, DEFAULT_OUTPUT_DIR, load_data, save_data
from text_cleaning import get_english_stopwords, text_to_words, text_to_words
from bow_features import extract_bow_features, train_word2vec_model
from cbow import MAX_VOCAB_SIZE, NUM_TRAIN_DOCS, WINDOW_SIZE, build_vocabulary, generate_context_target_pairs, most_similar, train_cbow


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


# def main() -> None:
#     args = parse_arguments()

#     dataset = load_data(dataset_name=args.dataset_name, output_dir=Path(args.output_dir))
#     print("Original dataset summary:")
#     print_dataset_summary(dataset)

#     dataset_preprocessed = preprocess_dataset(dataset = dataset, dataset_name=args.dataset_name, output_dir=Path(args.output_dir))
#     print("Preprocessed dataset summary:")
#     print_dataset_summary(dataset_preprocessed)

#     # print(f"\nSample cleaned texts (first {args.sample_count}):")
#     # for i in range(args.sample_count):
#     #     print("-----------------------------")
#     #     print(f"Original Text {i + 1}: {dataset_preprocessed['train'][i]['text']}")
#     #     print("\n")
#     #     print(f"Cleaned Text {i + 1}: {dataset_preprocessed['train'][i]['cleaned_text']}")

#     # w2v_model = train_word2vec_model(dataset_preprocessed, text_column="cleaned_text")
#     # print("Vocabulary size:", len(w2v_model.wv))
#     # print("Vector for 'good':", w2v_model.wv["good"][:5])

    
#     # print("Similarity between 'good' and 'great':", w2v_model.wv.similarity("good", "great"))

#     tokenized_docs = [doc.split() for doc in dataset_preprocessed["train"]["cleaned_text"] if doc]
#     word2idx, idx2word = build_vocabulary(tokenized_docs)
#     print("word2idx:", list(word2idx.items())[:10])  # Print first 10 entries for brevity
#     print("idx2word:", idx2word[:10])  # Print first 10 entries for brevity

#     context_target_pairs = generate_context_target_pairs(tokenized_docs, word2idx, window_size=2)
#     print("Context-Target pairs:", context_target_pairs[:10])  # Print first 10 pairs for brevity   


def test_build_vocabulary():
    # A tiny fake corpus - 3 "documents", already tokenized
    tokenized_docs = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "log"],
        ["cat", "and", "dog", "are", "friends"],
    ]

    word2idx, idx2word = build_vocabulary(
        tokenized_docs,
        min_count=2,      # keep everything for this small test
        max_vocab_size=100,
    )

    print("word2idx:", word2idx)
    print("idx2word:", idx2word)

    # # --- sanity checks ---
    # # "the" appears 4 times total - should be in the vocab
    # assert "the" in word2idx
    # # word2idx and idx2word should be inverses of each other
    # for word, idx in word2idx.items():
    #     assert idx2word[idx] == word
    # # no duplicate indices
    # assert len(word2idx) == len(set(word2idx.values()))

    # print("All checks passed. Vocab size:", len(word2idx))
    context_target_pairs = generate_context_target_pairs(tokenized_docs, word2idx, window_size=2)
    print("Context-Target pairs:", context_target_pairs[:10])  # Print first 10 pairs for brevity   


# if __name__ == "__main__":
#     #main()
#     test_build_vocabulary()


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    dataset = load_data(dataset_name=DEFAULT_DATASET_NAME, output_dir=output_dir)
    stop_words = get_english_stopwords(output_dir)

    train_split = dataset["train"]
    n_docs = min(NUM_TRAIN_DOCS, len(train_split))
    print(f"Cleaning {n_docs} documents...")

    tokenized_docs = []
    for i in range(n_docs):
        cleaned = text_to_words(train_split[i]["text"], stop_words)
        tokenized_docs.append(cleaned.split())
        print(f"Cleaned {i + 1}/{n_docs}", end="\r")
    print()

    word2idx, idx2word = build_vocabulary(tokenized_docs)
    print(f"Vocabulary size: {len(word2idx)}")

    pairs = generate_context_target_pairs(tokenized_docs, word2idx, WINDOW_SIZE)
    print(f"Generated {len(pairs)} (context, target) training pairs")

    model = train_cbow(pairs, vocab_size=len(word2idx))

    for probe_word in ["good", "bad", "movie"]:
        neighbors = most_similar(model, probe_word, word2idx, idx2word)
        print(f"Words similar to '{probe_word}': {neighbors}")


if __name__ == "__main__":
    main()

