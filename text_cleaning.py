import re
import nltk

from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional, Set

STOPWORDS_RESOURCE = "stopwords"
LETTER_ONLY_PATTERN = re.compile(r"[^a-zA-Z]")
DEFAULT_OUTPUT_DIR = Path("dataset")

def ensure_nltk_resource(resource_name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Download an NLTK resource if it is not already available in output_dir/nltk."""    
    local_nltk_path = Path(output_dir) / "nltk"
    # Create directory if it doesn't exist
    local_nltk_path.mkdir(parents=True, exist_ok=True)
    
    # Add to NLTK's search path
    nltk_path_str = str(local_nltk_path)
    if nltk_path_str not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path_str)
    
    # Check if resource is already available locally
    corpora_path = local_nltk_path / "corpora" / resource_name
    tokenizers_path = local_nltk_path / "tokenizers" / resource_name
    
    if corpora_path.exists() or tokenizers_path.exists():
        print(f"NLTK resource '{resource_name}' already exists locally at '{local_nltk_path}'.")
        return  # Resource already exists locally
    
    # Download and save to output_dir/nltk if not found
    nltk.download(resource_name, download_dir=nltk_path_str)


def get_english_stopwords(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Set[str]:
    """Return a set of English stopwords from NLTK."""
    ensure_nltk_resource(STOPWORDS_RESOURCE, output_dir)
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
