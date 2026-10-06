"""Training der Topic-Modelle und Gütemaße, gemeinsam genutzt von Schritt 4 und 5."""
import pickle

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from scipy import sparse
from sklearn.decomposition import NMF

from config import DATA, SEED, TOP_N


def load_vectors() -> tuple[Dictionary, list, sparse.csr_matrix, sparse.csr_matrix]:
    """Lädt Wörterbuch, BoW-Korpus, TF-IDF-Matrix und die BoW-Zählmatrix für NMF."""
    dictionary = Dictionary.load(str(DATA / "dictionary.gensim"))
    with open(DATA / "bow_corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    tfidf = sparse.load_npz(DATA / "tfidf.npz")
    rows = [i for i, doc in enumerate(corpus) for _ in doc]
    cols = [w for doc in corpus for w, _ in doc]
    vals = [c for doc in corpus for _, c in doc]
    counts = sparse.csr_matrix((vals, (rows, cols)), shape=tfidf.shape, dtype=float)
    return dictionary, corpus, tfidf, counts


def train_lda(corpus: list, dictionary: Dictionary, k: int) -> LdaModel:
    """Trainiert ein LDA-Modell mit gelernten Dirichlet-Prioren."""
    return LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=k, passes=10, iterations=100,
        chunksize=2000, alpha="auto", eta="auto", random_state=SEED, eval_every=None,
    )


def train_nmf(matrix: sparse.csr_matrix, k: int) -> NMF:
    """Trainiert ein NMF-Modell auf einer Dokument-Term-Matrix."""
    model = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=SEED)
    model.fit(matrix)
    return model


def lda_top_words(model: LdaModel, n: int = TOP_N) -> list[list[str]]:
    """Gibt die Top-Wörter je LDA-Thema zurück."""
    return [[w for w, _ in model.show_topic(t, topn=n)] for t in range(model.num_topics)]


def nmf_top_words(model: NMF, vocabulary: list[str], n: int = TOP_N) -> list[list[str]]:
    """Gibt die Top-Wörter je NMF-Thema zurück."""
    return [[vocabulary[i] for i in comp.argsort()[::-1][:n]] for comp in model.components_]


def coherence(topics: list[list[str]], texts: list[list[str]], dictionary: Dictionary) -> float:
    """Berechnet die Kohärenz c_v nach Röder et al. (2015) über alle Themen."""
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence="c_v", processes=1)
    return float(cm.get_coherence())


def diversity(topics: list[list[str]]) -> float:
    """Anteil eindeutiger Wörter unter allen Top-Wörtern (1 = keine Überschneidung)."""
    words = [w for t in topics for w in t]
    return len(set(words)) / len(words)


def dominant_topics(doc_topic: np.ndarray) -> np.ndarray:
    """Ordnet jedem Dokument das Thema mit dem höchsten Anteil zu."""
    return doc_topic.argmax(axis=1)
