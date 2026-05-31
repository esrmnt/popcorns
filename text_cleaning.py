import re
import nltk

from bs4 import BeautifulSoup
from typing import Optional, Set

STOPWORDS_RESOURCE = "stopwords"
LETTER_ONLY_PATTERN = re.compile(r"[^a-zA-Z]")

def ensure_nltk_resource(resource_name: str) -> None:
    """Download an NLTK resource if it is not already installed."""
    try:
        nltk.data.find(f"corpora/{resource_name}")
    except LookupError:
        nltk.download(resource_name, download_dir="corpora/{resource_name}")


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
