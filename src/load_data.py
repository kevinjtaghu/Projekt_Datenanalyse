"""Schritt 1: Rohdaten laden, filtern, deduplizieren und eine Stichprobe ziehen."""
import pandas as pd

from config import RAW_CSV, SAMPLE_FILE, SAMPLE_SIZE, SEED, TABLES, YEAR

COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Consumer complaint narrative": "narrative",
}


def read_year() -> tuple[pd.DataFrame, int]:
    """Liest die große CSV in Blöcken und behält nur Beschwerden mit Freitext aus YEAR."""
    parts, total = [], 0
    for chunk in pd.read_csv(RAW_CSV, usecols=list(COLUMNS), dtype=str, chunksize=500_000):
        total += len(chunk)
        chunk = chunk.rename(columns=COLUMNS).dropna(subset=["narrative"])
        parts.append(chunk[chunk["date_received"].str.startswith(YEAR)])
    return pd.concat(parts, ignore_index=True), total


def main() -> None:
    """Erzeugt die Stichprobe und eine Tabelle zur Datenqualität."""
    if not RAW_CSV.exists():
        raise SystemExit(f"Rohdatei fehlt: {RAW_CSV} (Download siehe README)")
    df, total_rows = read_year()
    with_narrative = len(df)

    key = df["narrative"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.loc[~key.duplicated()].copy()
    after_dedup = len(df)

    df["n_words"] = df["narrative"].str.split().str.len()
    df = df[df["n_words"] >= 20]
    after_length = len(df)

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    sample.drop(columns="n_words").to_csv(SAMPLE_FILE, index=False, compression="gzip")

    quality = pd.DataFrame(
        {
            "kennzahl": [
                "Zeilen gesamt (alle Jahre)",
                f"mit Freitext in {YEAR}",
                "nach Entfernen exakter Duplikate",
                "nach Entfernen sehr kurzer Texte (< 20 Wörter)",
                "Stichprobe",
                "Median Wörter je Text (Stichprobe)",
            ],
            "wert": [
                total_rows,
                with_narrative,
                after_dedup,
                after_length,
                len(sample),
                int(sample["n_words"].median()),
            ],
        }
    )
    quality.to_csv(TABLES / "data_quality.csv", index=False)
    (sample["product"].value_counts(normalize=True).round(3)
     .rename("anteil").to_csv(TABLES / "sample_products.csv"))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
