"""
CBOW (Continuous Bag of Words) implementation.
"""

import random
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

EPOCHS = 50
WINDOW_SIZE = 2
EMBEDDING_DIM = 50

LEARNING_RATE = 0.05
NUM_TRAIN_DOCS = 500        
MIN_WORD_COUNT = 2          

MAX_VOCAB_SIZE = 10_000   

def build_vocabulary(
    tokenized_docs: List[List[str]],
    min_count: int = MIN_WORD_COUNT,
    max_vocab_size: int = MAX_VOCAB_SIZE,
) -> Tuple[Dict[str, int], List[str]]:
    """Build word2idx / idx2word from a list of tokenized documents.

    Args:
        tokenized_docs: list of documents, each a list of word tokens.
        min_count: drop words that occur fewer than this many times.
        max_vocab_size: keep only the most frequent `max_vocab_size` words.

    Returns:
        word2idx: mapping from word to integer index
        idx2word: list where idx2word[i] is the word for index i
    """

    words = []
    for doc in tokenized_docs:
        for word in doc:
            words.append(word)

    counts = Counter(words)

    most_common = [word for word, count in counts.most_common(max_vocab_size) if count >= min_count]

    idx2word = most_common
    word2idx = {word: idx for idx, word in enumerate(idx2word)}

    return word2idx, idx2word

def generate_context_target_pairs(
    tokenized_docs: List[List[str]],
    word2idx: Dict[str, int],
    window_size: int = WINDOW_SIZE,
) -> List[Tuple[List[int], int]]:
    """Turn tokenized documents into (context_indices, target_index) pairs.

    For each word in each document (that's in the vocabulary), gather up to `window_size` words on each side (also filtered to the vocabulary) as context, and use the word itself as the target.

    Words not in word2idx are skipped entirely (both as context and as target) - this is the simplest way to handle OOV words for a from - scratch implementation.

    Returns:
        A list of (context_indices, target_index) tuples. context_indices
        may have variable length near document edges.
    """
    pairs = []

    for doc in tokenized_docs:
        indices = [word2idx[w] for w in doc if w in word2idx]
        
        for i, target_idx in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            context_idx = [indices[j] for j in range(start, end) if j != i]
            if context_idx:
                pairs.append((context_idx, target_idx))
            
    return pairs

class CBOW:
    """A from-scratch CBOW model with a single hidden (embedding) layer.

    Weight shapes:
        W_in:  (vocab_size, embedding_dim)  - input/embedding lookup table.
               W_in[word_idx] is the embedding vector for that word.
        W_out: (embedding_dim, vocab_size)  - output projection.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)
        # Small random init. Common choice: uniform(-0.5, 0.5) / embedding_dim, ~ original word2vec C code. 
        self.W_in = (rng.random((vocab_size, embedding_dim)) - 0.5) / embedding_dim
        self.W_out = np.zeros((embedding_dim, vocab_size))

    def forward(self, context_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Run the forward pass for one training example.

        Args:
            context_indices: list of word indices in the context window.

        Returns:
            h: the hidden layer, i.e. the averaged context embedding.
               shape (embedding_dim,)
            y_hat: predicted probability distribution over the vocabulary.
               shape (vocab_size,)
        """
        context_embeddings = self.W_in[context_indices]  
        h = np.mean(context_embeddings, axis=0)  
        logits = h @ self.W_out
        y_hat = softmax(logits)

        return h, y_hat
    
    def compute_loss(self, y_hat: np.ndarray, target_idx: int) -> float:
        """Cross-entropy loss for a single example.

        Args:
            y_hat: predicted probability distribution over the vocabulary.
            target_idx: the index of the true target word. 
        
        Returns:
            loss: the cross-entropy loss for this example.
        """
        epsilon = 1e-10
        loss = -np.log(y_hat[target_idx] + epsilon)
        return loss

    def backward_and_update(
        self,
        context_indices: List[int],
        target_idx: int,
        h: np.ndarray,
        y_hat: np.ndarray,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        """Compute gradients for one example and apply an SGD update in place.

        Args:
            context_indices: list of word indices in the context window.
            target_idx: the index of the true target word.
            h: the hidden layer (averaged context embedding).
            y_hat: predicted probability distribution over the vocabulary.
            learning_rate: the SGD learning rate.
        
        Note: this is the full softmax CBOW gradient - no negative sampling or hierarchical softmax.
        """
        y = np.zeros(self.vocab_size)
        y[target_idx] = 1

        e = y_hat - y
        dW_out = np.outer(h, e)  
        EH = self.W_out @ e  
        self.W_out -= learning_rate * dW_out
        for idx in context_indices:
            self.W_in[idx] -= learning_rate * (EH / len(context_indices))

    def get_word_vector(self, word_idx: int) -> np.ndarray:
        """Return the learned embedding for a given word index."""
        return self.W_in[word_idx]


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax.

    Args:
        x: input array of shape (vocab_size,)
    Returns:
        softmaxed: array of shape (vocab_size,) representing a probability distribution.
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def train_cbow(
    pairs: List[Tuple[List[int], int]],
    vocab_size: int,
    embedding_dim: int = EMBEDDING_DIM,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> CBOW:
    """Train a CBOW model over the given (context, target) pairs.

    Args:
        pairs: list of (context_indices, target_index) tuples.
        vocab_size: size of the vocabulary (number of unique words).
        embedding_dim: size of the embedding vectors.
        epochs: number of passes over the training data.
        learning_rate: SGD learning rate.
    """
    model = CBOW(vocab_size=vocab_size, embedding_dim=embedding_dim)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0

        for i, (context_indices, target_idx) in enumerate(pairs):
            h, y_hat = model.forward(context_indices)
            loss = model.compute_loss(y_hat, target_idx)
            model.backward_and_update(context_indices, target_idx, h, y_hat, learning_rate)

            total_loss += loss
            if (i + 1) % 1000 == 0:
                print(f"epoch {epoch + 1}/{epochs} - example {i + 1}/{len(pairs)} - avg loss {total_loss / (i + 1):.4f}", end="\r")

        print(f"\nEpoch {epoch + 1}/{epochs} complete - avg loss: {total_loss / len(pairs):.4f}")

    return model

def most_similar(model: CBOW, word: str, word2idx: Dict[str, int], idx2word: List[str], top_n: int = 5):
    """Return the top_n words most similar (cosine similarity) to `word`."""
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return []

    vec = model.get_word_vector(word2idx[word])
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)

    all_vecs = model.W_in
    all_norms = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-10)
    sims = all_norms @ vec_norm

    top_indices = np.argsort(-sims)[: top_n + 1]  # +1 to skip the word itself
    return [(idx2word[i], sims[i]) for i in top_indices if idx2word[i] != word][:top_n]

