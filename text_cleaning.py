import re
import nltk

from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional, Set

STOPWORDS_RESOURCE = "stopwords"
LETTER_ONLY_PATTERN = re.compile(r"[^a-zA-Z]")
DEFAULT_LOCAL_NLTK_PATH = Path("dataset/nltk")

def ensure_nltk_resource(resource_name: str) -> None:
    """Download an NLTK resource if it is not already available in dataset/nltk."""    
    # Create directory if it doesn't exist
    DEFAULT_LOCAL_NLTK_PATH.mkdir(parents=True, exist_ok=True)
    
    # Add to NLTK's search path
    if DEFAULT_LOCAL_NLTK_PATH not in nltk.data.path:
        nltk.data.path.insert(0, DEFAULT_LOCAL_NLTK_PATH)
    
    # Check if resource is already available
    try:
        nltk.data.find(resource_name)
    except LookupError:
        # Download and save to dataset/nltk if not found
        nltk.download(resource_name, download_dir=DEFAULT_LOCAL_NLTK_PATH)


def get_english_stopwords() -> Set[str]:
    """Return a set of English stopwords from NLTK."""
    ensure_nltk_resource(STOPWORDS_RESOURCE)
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


def review_to_words(raw_review: str, stop_words: Optional[Set[str]] = None) -> str:
    """Clean a raw review string and return a space-separated sequence of words."""
    if stop_words is None:
        stop_words = get_english_stopwords()

    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = LETTER_ONLY_PATTERN.sub(" ", review_text)
    words = letters_only.lower().split()
    meaningful_words = [word for word in words if word not in stop_words]
    return " ".join(meaningful_words)
