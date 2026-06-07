from datasets import DatasetDict
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer


def train_word2vec_model(dataset_name: DatasetDict, text_column: str = "cleaned_text", vector_size: int = 100,
                         window: int = 5, min_count: int = 1, workers: int = 1, seed: int = 42) -> Word2Vec:
    """Train a Word2Vec model on the cleaned review text."""
    sentences = [[token for token in str(text).split()] for text in dataset_name["train"][text_column] if str(text).split()]

    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )


def extract_bow_features(dataset_name: DatasetDict, text_column: str = "cleaned_text") -> DatasetDict:
    """Extract Bag-of-Words features from the cleaned text column."""
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words=None,
        tokenizer=None,
        analyzer="word",
        preprocessor=None,
    )
    train_texts = dataset_name["train"][text_column]
    bow_features = vectorizer.fit_transform(train_texts)

    bow_features_dense = bow_features.toarray().tolist()
    dataset_name["train"] = dataset_name["train"].add_column("bow_features", bow_features_dense)

    return dataset_name