import shutil
from pathlib import Path

from datasets import DatasetDict
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_MODEL_NAME = "word2vec.model"
DEFAULT_OUTPUT_DIR = Path("models")

def train_word2vec_model(dataset_name: DatasetDict, text_column: str = "cleaned_text", vector_size: int = 100,
                         window: int = 5, min_count: int = 1, workers: int = 1, seed: int = 42) -> Word2Vec:
    """Train a Word2Vec model on the cleaned review text."""
    """load model from disk if it exists, otherwise train a new model and save it to disk"""

    model_path = DEFAULT_OUTPUT_DIR / DEFAULT_MODEL_NAME
    
    if model_path.exists():
        if model_path.is_file():
            print(f"Loading existing Word2Vec model from {model_path}...")
            return Word2Vec.load(str(model_path))

        print(f"Removing invalid model path {model_path} before retraining...")
        shutil.rmtree(model_path, ignore_errors=True)
    
    print("Training new Word2Vec model...")
    sentences = [[token for token in str(text).split()] for text in dataset_name["train"][text_column] if str(text).split()]

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )

    save_model(model, model_path)
    return model

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

def save_model(model: Word2Vec, output_path: Path) -> None:
    """Save the trained Word2Vec model to the specified file path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
