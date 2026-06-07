# popcorns
A machine learning project exploring sentiment analysis on movie reviews using Word2Vec algorithm. This repository explores if distributed word embeddings can capture semantic relationships between words and improve text classification performance.


### Setup
```python
py -3.12 -m venv .venv 
.\.venv\Scripts\Activate.ps1
pip install pandas
pip install datasets 
pip install BeautifulSoup4
pip install nltk   
pip install scikit-learn
pip install gensim
```

### Dataset
https://huggingface.co/datasets/stanfordnlp/imdb

### Assumptions
- You are already logged in to the hugging face, if not follow the instructions here to login to be able to download the dataset
```html
https://huggingface.co/docs/huggingface_hub/en/quick-start
```