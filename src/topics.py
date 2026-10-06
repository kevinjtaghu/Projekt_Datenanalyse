"""Schritt 5: Finale LDA- und NMF-Modelle mit dem gewählten k trainieren und Themen exportieren."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.matutils import corpus2dense

from config import DATA, FIGURES, RESULTS, TABLES, load_clean, load_json
from models import (coherence, dominant_topics, lda_top_words, load_vectors, nmf_top_words,
                    train_lda, train_nmf)


def topic_table(name: str, topics: list[list[str]], weights: list[list[float]], assigned: np.ndarray) -> pd.DataFrame:
    """Baut eine Tabelle mit Top-Wörtern und Anteil der Dokumente je Thema."""
    share = np.bincount(assigned, minlength=len(topics)) / len(assigned)
    return pd.DataFrame({
        "modell": name,
        "thema": range(len(topics)),
        "anteil_dokumente": share.round(3),
        "top_woerter": [", ".join(t) for t in topics],
        "gewichte": [", ".join(f"{w:.3f}" for w in ws) for ws in weights],
    }).sort_values("anteil_dokumente", ascending=False)


def plot_topics(name: str, topics: list[list[str]], weights: list[list[float]], share: np.ndarray, filename: str) -> None:
    """Zeichnet die Top 8 Wörter je Thema als kleine Balkendiagramme."""
    order = np.argsort(share)[::-1]
    cols = 4
    rows = int(np.ceil(len(topics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, t in zip(axes.ravel(), order):
        ax.axis("on")
        ax.barh(topics[t][:8][::-1], weights[t][:8][::-1], color="#4C72B0")
        ax.set_title(f"Thema {t} ({share[t]:.0%} der Texte)", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle(name, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    """Trainiert die finalen Modelle, speichert Themen, Zuordnungen und Abbildungen."""
    df = load_clean()
    texts = df["tokens"].tolist()
    dictionary, corpus, tfidf, _ = load_vectors()
    vocabulary = [dictionary[i] for i in range(len(dictionary))]
    chosen = load_json(RESULTS / "chosen_k.json")

    k_lda = chosen["LDA (BoW)"]
    lda = train_lda(corpus, dictionary, k_lda)
    lda.save(str(DATA / "lda.gensim"))
    lda_topics = lda_top_words(lda)
    lda_weights = [[p for _, p in lda.show_topic(t, topn=10)] for t in range(k_lda)]
    lda_doc_topic = corpus2dense(lda.get_document_topics(corpus, minimum_probability=0), k_lda).T
    lda_assigned = dominant_topics(lda_doc_topic)

    k_nmf = chosen["NMF (TF-IDF)"]
    nmf = train_nmf(tfidf, k_nmf)
    nmf_topics = nmf_top_words(nmf, vocabulary)
    nmf_weights = [sorted(c, reverse=True)[:10] for c in nmf.components_]
    nmf_doc_topic = nmf.transform(tfidf)
    nmf_assigned = dominant_topics(nmf_doc_topic)

    tables = pd.concat([
        topic_table(f"LDA (k={k_lda})", lda_topics, lda_weights, lda_assigned),
        topic_table(f"NMF (k={k_nmf})", nmf_topics, nmf_weights, nmf_assigned),
    ])
    tables.to_csv(TABLES / "topics.csv", index=False)

    plot_topics(f"LDA auf Bag-of-Words, k = {k_lda}", lda_topics, lda_weights,
                np.bincount(lda_assigned, minlength=k_lda) / len(df), "topics_lda.png")
    plot_topics(f"NMF auf TF-IDF, k = {k_nmf}", nmf_topics, nmf_weights,
                np.bincount(nmf_assigned, minlength=k_nmf) / len(df), "topics_nmf.png")

    assignments = df[["complaint_id", "product", "issue"]].copy()
    assignments["lda_topic"] = lda_assigned
    assignments["lda_confidence"] = lda_doc_topic.max(axis=1).round(3)
    assignments["nmf_topic"] = nmf_assigned
    assignments.to_csv(TABLES / "document_topics.csv.gz", index=False, compression="gzip")

    print(f"LDA k={k_lda}: c_v={coherence(lda_topics, texts, dictionary):.3f}")
    print(f"NMF k={k_nmf}: c_v={coherence(nmf_topics, texts, dictionary):.3f}")
    print(tables[["modell", "thema", "anteil_dokumente", "top_woerter"]].to_string(index=False))


if __name__ == "__main__":
    main()
