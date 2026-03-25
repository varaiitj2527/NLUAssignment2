# NLU ASSIGNMENT 2



---

## Project Structure

| File | Description |
| :--- | :--- |
| **`buildcorpus.py`** | Scrapes IIT Jodhpur web addresses, cleans navigational junk, standardizes academic degrees, and tokenizes text into `corpus.txt`. |
| **`word2vec.py`** | Implements Skip-gram and CBOW with Negative Sampling from scratch (using analytical gradients) and compares them with Gensim implementations. |
| **`namegeneration.py`** | Features custom implementations of Vanilla RNN, Bidirectional LSTM, and Causal Attention models to generate new text sequences. |

---

## Requirements & Installation

Before running the scripts, install the necessary dependencies using `pip`:

```bash
pip install beautifulsoup4 requests nltk wordcloud gensim scikit-learn numpy torch matplotlib
```

* **Note**: The scripts utilize `nltk` resources like `punkt` and `punkt_tab`, which are automatically downloaded upon running the corpus builder.

---

## Execution Instructions

### 1. Corpus Building
Run this script first to create the dataset required for the other models.
```bash
python buildcorpus.py
```
* **Output**: Generates `corpus.txt` and displays a Word Cloud of the most frequent academic terms.

### 2. Word Embedding Training (Word2Vec)
This script evaluates semantic relationships within the scraped data.
```bash
python word2vec.py
```
* **Input**: When prompted, enter the path to the generated file (e.g., `corpus.txt`).
* **Output**: Saves T-SNE visualizations (`tsne_scratch.png`, `tsne_gensim.png`) showing clusters for "Academic", "Research", and "AI".

### 3. Sequence Generation
This script trains generative models on the corpus.
```bash
python namegeneration.py
```
* **Input**: Provide the path to `corpus.txt`.
* **Output**: Calculates **Novelty Rate** and **Diversity** metrics and saves generated results to `.txt` files (e.g., `VanillaRnn_generated.txt`).

---
