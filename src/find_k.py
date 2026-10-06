"""Schritt 4: Optimale Themenanzahl k über Kohärenz und Diversität bestimmen."""
import time

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES, K_RANGE, RESULTS, TABLES, load_clean, save_json
from models import (coherence, diversity, lda_top_words, load_vectors, nmf_top_words,
                    train_lda, train_nmf)

MODELS = ["LDA (BoW)", "NMF (TF-IDF)", "NMF (BoW)"]


def evaluate_k(k, dictionary, corpus, tfidf, counts, vocabulary, texts) -> list[dict]:
    """Trainiert alle drei Varianten für ein k und berechnet die Gütemaße."""
    rows = []
    lda = train_lda(corpus, dictionary, k)
    variants = {
        "LDA (BoW)": (lda_top_words(lda), {"perplexity_bound": lda.log_perplexity(corpus)}),
        "NMF (TF-IDF)": (nmf_top_words(train_nmf(tfidf, k), vocabulary), {}),
        "NMF (BoW)": (nmf_top_words(train_nmf(counts, k), vocabulary), {}),
    }
    for name, (topics, extra) in variants.items():
        rows.append({"modell": name, "k": k, "coherence_cv": coherence(topics, texts, dictionary),
                     "diversity": diversity(topics), **extra})
    return rows


def choose_k(results: pd.DataFrame) -> dict:
    """Wählt je Modell das k mit maximaler Kohärenz unter allen k mit Diversität von mindestens 0,7."""
    chosen = {}
    for name, grp in results.groupby("modell"):
        candidates = grp[grp["diversity"] >= 0.7]
        if candidates.empty:
            candidates = grp
        best = candidates.loc[candidates["coherence_cv"].idxmax()]
        chosen[name] = int(best["k"])
    return chosen


def plot(results: pd.DataFrame, chosen: dict) -> None:
    """Zeichnet Kohärenz und Diversität je k für alle Modelle."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in MODELS:
        grp = results[results["modell"] == name]
        line, = axes[0].plot(grp["k"], grp["coherence_cv"], marker="o", label=name)
        axes[0].axvline(chosen[name], color=line.get_color(), linestyle=":", alpha=0.7)
        axes[1].plot(grp["k"], grp["diversity"], marker="o", label=name)
    axes[0].set(title="Kohärenz c_v (höher ist besser)", xlabel="Anzahl Themen k", ylabel="c_v")
    axes[1].set(title="Themen-Diversität (Top 10 Wörter)", xlabel="Anzahl Themen k", ylabel="Anteil eindeutiger Wörter")
    axes[1].axhline(0.7, color="grey", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xticks(K_RANGE)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "k_selection.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Durchläuft K_RANGE, speichert die Kennzahlen und die gewählten k."""
    texts = load_clean()["tokens"].tolist()
    dictionary, corpus, tfidf, counts = load_vectors()
    vocabulary = [dictionary[i] for i in range(len(dictionary))]

    rows = []
    for k in K_RANGE:
        start = time.time()
        rows += evaluate_k(k, dictionary, corpus, tfidf, counts, vocabulary, texts)
        print(f"k={k:2d} fertig ({time.time() - start:.0f}s)", flush=True)

    results = pd.DataFrame(rows)
    results.round(4).to_csv(TABLES / "k_selection.csv", index=False)
    chosen = choose_k(results)
    save_json(chosen, RESULTS / "chosen_k.json")
    plot(results, chosen)
    print(results.pivot(index="k", columns="modell", values="coherence_cv").round(3))
    print("Gewählt:", chosen)


if __name__ == "__main__":
    main()
