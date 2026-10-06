"""Schritt 6: Gefundene Themen mit den behördlichen Produkt-Labels abgleichen."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import FIGURES, TABLES

SHORT_PRODUCT = {
    "Debt collection": "Debt collection",
    "Checking or savings account": "Checking/savings",
    "Credit card": "Credit card",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Mortgage": "Mortgage",
    "Vehicle loan or lease": "Vehicle loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Student loan": "Student loan",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Prepaid card": "Prepaid card",
    "Debt or credit management": "Debt management",
}


def heatmap(df: pd.DataFrame, column: str, title: str, filename: str) -> pd.DataFrame:
    """Zeigt je Thema, wie sich seine Dokumente auf die Produkte verteilen."""
    table = pd.crosstab(df[column], df["product_short"], normalize="index")
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(table) + 2))
    im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)), [f"Thema {t}" for t in table.index], fontsize=8)
    for (i, j), v in pd.DataFrame(table.values).stack().items():
        if v >= 0.1:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)
    return table


def main() -> None:
    """Berechnet NMI und ARI gegen die Produkt- und Issue-Labels und erstellt Heatmaps."""
    df = pd.read_csv(TABLES / "document_topics.csv.gz")
    df["product_short"] = df["product"].map(SHORT_PRODUCT).fillna(df["product"])

    rows = []
    for column, name in [("lda_topic", "LDA"), ("nmf_topic", "NMF")]:
        for label in ["product", "issue"]:
            rows.append({
                "modell": name, "label": label,
                "nmi": round(normalized_mutual_info_score(df[label], df[column]), 3),
                "ari": round(adjusted_rand_score(df[label], df[column]), 3),
            })
        table = heatmap(df, column, f"{name}: Verteilung der Produkte je Thema", f"topics_vs_product_{name.lower()}.png")
        purity = table.max(axis=1).rename("reinheit").to_frame()
        purity["haupt_produkt"] = table.idxmax(axis=1)
        purity.round(3).to_csv(TABLES / f"topic_purity_{name.lower()}.csv")

    scores = pd.DataFrame(rows)
    scores.to_csv(TABLES / "label_agreement.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
