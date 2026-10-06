"""Führt die komplette Pipeline aus; Schritt 1 entfällt, wenn die Stichprobe schon vorliegt."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
SAMPLE = Path(__file__).resolve().parent / "data" / "sample_2026.csv.gz"
STEPS = ["load_data", "preprocessing", "vectorization", "find_k", "topics", "evaluation"]

if __name__ == "__main__":
    for step in STEPS:
        if step == "load_data" and SAMPLE.exists():
            print("Stichprobe vorhanden, Schritt 1 wird übersprungen.")
            continue
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(SRC / f"{step}.py")], check=True, cwd=SRC)
