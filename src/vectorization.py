"""Schritt 3: Bag-of-Words (gensim) und TF-IDF (scikit-learn) auf gleichem Vokabular erzeugen und vergleichen."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA, FIGURES, NO_ABOVE, NO_BELOW, TABLES, load_clean


def build_bow(tokens: list[list[str]]) -> tuple[Dictionary, list]:
    """Erstellt Wörterbuch und Bag-of-Words-Korpus und entfernt seltene sowie zu häufige Wörter."""
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=None)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]
    return dictionary, corpus


def build_tfidf(texts: pd.Series, vocabulary: list[str]) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Erstellt die TF-IDF-Matrix mit exakt demselben Vokabular wie der BoW-Korpus."""
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(texts)


def plot_top_terms(bow_top: pd.Series, tfidf_top: pd.Series) -> None:
    """Stellt die wichtigsten Begriffe beider Verfahren nebeneinander dar."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, series, title in [
        (axes[0], bow_top, "Bag-of-Words: absolute Häufigkeit"),
        (axes[1], tfidf_top, "TF-IDF: mittleres Gewicht"),
    ]:
        series[::-1].plot.barh(ax=ax, color="#4C72B0")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / "vectorization_top_terms.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Vektorisiert die Texte, speichert beide Repräsentationen und vergleicht sie."""
    df = load_clean()
    dictionary, corpus = build_bow(df["tokens"])
    vocabulary = [dictionary[i] for i in range(len(dictionary))]
    vectorizer, tfidf = build_tfidf(df["clean_text"], vocabulary)

    dictionary.save(str(DATA / "dictionary.gensim"))
    with open(DATA / "bow_corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    sparse.save_npz(DATA / "tfidf.npz", tfidf)

    counts = pd.Series({dictionary[i]: c for i, c in dictionary.cfs.items()})
    bow_top = counts.sort_values(ascending=False).head(20)
    tfidf_mean = pd.Series(np.asarray(tfidf.mean(axis=0)).ravel(), index=vocabulary)
    tfidf_top = tfidf_mean.sort_values(ascending=False).head(20)
    plot_top_terms(bow_top, tfidf_top)

    n_docs, n_terms = tfidf.shape
    comparison = pd.DataFrame({
        "kennzahl": ["Dokumente", "Vokabular", "Dichte der Matrix", "Ø verschiedene Wörter je Text"],
        "wert": [n_docs, n_terms, round(tfidf.nnz / (n_docs * n_terms), 4), round(tfidf.nnz / n_docs, 1)],
    })
    comparison.to_csv(TABLES / "vectorization_stats.csv", index=False)
    pd.DataFrame({
        "rang": range(1, 21),
        "bow_begriff": bow_top.index, "bow_haeufigkeit": bow_top.values,
        "tfidf_begriff": tfidf_top.index, "tfidf_gewicht": tfidf_top.values.round(4),
    }).to_csv(TABLES / "vectorization_top_terms.csv", index=False)

    overlap = len(set(bow_top.index) & set(tfidf_top.index))
    print(comparison.to_string(index=False))
    print(f"Überschneidung Top 20: {overlap}/20")
    print("BoW:  ", ", ".join(bow_top.index))
    print("TFIDF:", ", ".join(tfidf_top.index))


if __name__ == "__main__":
    main()
