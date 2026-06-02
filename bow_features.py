from sklearn.feature_extraction.text import CountVectorizer
from datasets import DatasetDict

def extract_bow_features(dataset: DatasetDict, text_column: str = "cleaned_text") -> DatasetDict:
    """Extract Bag-of-Words features from the cleaned text column."""
    vectorizer = CountVectorizer(
        max_features=5000,  # Limit to top 5000 words
        stop_words=None,  # Remove English stop words
        tokenizer=None,  # Tokenize by words
        analyzer="word",
        preprocessor=None
    )
    train_texts = dataset["train"][text_column]
    bow_features = vectorizer.fit_transform(train_texts)
    
    # Convert the sparse matrix to a dense format and then to a list of lists
    bow_features_dense = bow_features.toarray().tolist()
    
    # Add the BoW features as a new column in the dataset
    dataset["train"] = dataset["train"].add_column("bow_features", bow_features_dense)
    
    return dataset