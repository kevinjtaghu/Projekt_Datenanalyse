"""Gemeinsame Pfade, Parameter und Hilfsfunktionen der Pipeline."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT.parent / "complaints" / "complaints.csv"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

SAMPLE_FILE = DATA / "sample_2026.csv.gz"
CLEAN_FILE = DATA / "clean_2026.parquet"

YEAR = "2026"
SAMPLE_SIZE = 20_000
SEED = 42

NO_BELOW = 10
NO_ABOVE = 0.5
K_RANGE = list(range(4, 25, 2))
TOP_N = 10

for folder in (DATA, FIGURES, TABLES):
    folder.mkdir(parents=True, exist_ok=True)


def load_clean() -> pd.DataFrame:
    """Lädt die vorverarbeiteten Texte inklusive Token-Listen."""
    df = pd.read_parquet(CLEAN_FILE)
    df["tokens"] = df["clean_text"].str.split()
    return df


def save_json(obj, path: Path) -> None:
    """Speichert ein Objekt lesbar als JSON."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    """Lädt eine JSON-Datei."""
    return json.loads(path.read_text(encoding="utf-8"))
