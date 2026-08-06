# popcorns

popcorns is a small research-oriented codebase that explores learning word embeddings from movie reviews and inspecting their semantic properties. The implementation trains a simple continious bag of words model on an IMDb dataset.

What this repo contains
- Data loading and disk cache logic using the Hugging Face `datasets` library.
- Text cleaning utilities that use BeautifulSoup and NLTK stopwords.
- A CBOW implementation witha a simple training loop.
- An `orchestrator.py` script that ties the steps together: load, preprocess, build vocabulary, generate training pairs, train CBOW, and print nearest neighbours for probe words.

Repository layout
- `orchestrator.py`: end-to-end runner for data preprocessing and CBOW training.
- `data_loader.py`: download / load and save datasets to `dataset/`.
- `text_cleaning.py`: HTML stripping, tokenization and NLTK stopword management.
- `cbow.py`: CBOW model, training loop and helper functions.
- `dataset/`: local dataset cache and NLTK resources (created at runtime).

Requirements
- Python 3.12 recommended.
- Key Python packages: `numpy`, `datasets`, `beautifulsoup4`, `nltk`.

Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy datasets beautifulsoup4 nltk
```

#### Running the project
1. By default the project uses the `stanfordnlp/imdb` dataset from Hugging Face. Running the orchestrator will download the dataset (if not cached), ensure NLTK stopwords are available under `dataset/nltk`, preprocess texts and train a small CBOW model on a configurable number of documents.

```bash
python3 orchestrator.py
```

Command-line options for `orchestrator.py`
- `--dataset-name`: Hugging Face dataset id (default: set in `data_loader.py`).
- `--output-dir`: directory where the dataset and preprocessing artifacts are stored (default: `dataset`).
- `--sample-count`: number of sample documents to print (used for quick inspection).

Notes and implementation details
- The text cleaning pipeline strips HTML, keeps only letters, lowercases and removes NLTK stopwords. NLTK resources are stored and looked up under `dataset/nltk` so the repository can work offline once resources are downloaded.
- The CBOW implementation in `cbow.py` is a didactic, from-scratch model using a full-softmax output. Training can be slow for large vocabularies; consider reducing `NUM_TRAIN_DOCS` or `EPOCHS` (defined in `cbow.py`) for quick experiments.
- Datasets downloaded via Hugging Face are saved with `save_to_disk` under `dataset/<dataset_name_with_underscores>/`.

License
- See the `LICENSE` file in the repository root.

#### References

Research papers
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. https://arxiv.org/abs/1310.4546
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/pubs/glove.pdf
- Goldberg, Y., & Levy, O. (2014). word2vec Explained: deriving Mikolov et al.'s negativesampling word-embedding method. https://arxiv.org/abs/1402.3722

Implementations & libraries
- Gensim — A mature Python library for topic modelling and vector representations (includes word2vec implementations). https://github.com/RaRe-Technologies/gensim
- Original `word2vec` C implementation (reference implementation and code examples). https://github.com/tmikolov/word2vec

Tutorials & explanatory articles
- The Illustrated Word2vec — an approachable, visual explanation of how word2vec works. https://jalammar.github.io/illustrated-word2vec/
- TensorFlow `word2vec` tutorial — a hands-on walkthrough with code. https://www.tensorflow.org/tutorials/text/word2vec
- Gensim word2vec examples and tutorials. https://radimrehurek.com/gensim/auto_examples/index.html

Datasets
- IMDb reviews on Hugging Face (used by this project): https://huggingface.co/datasets/stanfordnlp/imdb
