# Themenextraktion aus Verbraucherbeschwerden (NLP)

Projekt im Kurs **DLBDSEDA02 Projekt: Data Analysis** (IU Internationale Hochschule), Aufgabe 1.
Ziel ist es, die am häufigsten angesprochenen Themen aus unstrukturierten Beschwerdetexten
automatisch zu extrahieren und für Entscheidungstragende verdichtet darzustellen.

## Datenquelle

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
des US Consumer Financial Protection Bureau (Public Domain). Verwendet wird das Freitextfeld
`Consumer complaint narrative`; die behördlichen Labels `Product` und `Issue` dienen nur zur Validierung.

Die Analyse nutzt eine Zufallsstichprobe von 20.000 Beschwerden aus dem Jahr 2026 (aktuelle Themen,
ausgewogenerer Produktmix als 2025). Die Stichprobe liegt bereits unter `data/sample_2026.csv.gz`,
der Download der Gesamtdatei (ca. 9 GB) ist daher nur für Schritt 1 nötig.

## Pipeline

| Schritt | Datei in `src/` | Inhalt | Bibliotheken |
|---|---|---|---|
| 1 | `load_data.py` | Rohdaten blockweise laden, Jahr 2026 filtern, Duplikate und sehr kurze Texte entfernen, Stichprobe ziehen | pandas |
| 2 | `preprocessing.py` | Platzhalter (`XXXX`), Beträge, Links, Zahlen und Satzzeichen entfernen, Kleinschreibung, Tokenisierung, Lemmatisierung, Stoppwörter, Bigramme | spaCy, nltk, gensim |
| 3 | `vectorization.py` | Bag-of-Words und TF-IDF auf identischem Vokabular, Vergleich der wichtigsten Begriffe | gensim, scikit-learn |
| 4 | `find_k.py` | Bestimmung der optimalen Themenanzahl k (siehe unten) | gensim, scikit-learn, matplotlib |
| 5 | `topics.py` | Finale Modelle LDA (auf BoW) und NMF (auf TF-IDF), Themen und Zuordnungen exportieren | gensim, scikit-learn |
| 6 | `evaluation.py` | Abgleich der Themen mit den Produkt-Labels (NMI, ARI, Heatmaps) | scikit-learn, matplotlib |

### Wahl der Themenanzahl k

Für k = 4, 6, ..., 24 werden drei Varianten trainiert: LDA auf Bag-of-Words, NMF auf TF-IDF und
zum Vergleich der Vektorisierungen NMF auf Bag-of-Words. Je Modell werden berechnet:

* **Kohärenz c_v** (Röder, Both und Hinneburg 2015): misst, wie häufig die Top-Wörter eines Themas
  gemeinsam in den Texten auftreten, und korreliert gut mit menschlicher Beurteilung.
* **Themen-Diversität**: Anteil eindeutiger Wörter unter den Top 10 Wörtern aller Themen. Werte unter
  0,7 deuten auf redundante Themen hin.
* Für LDA zusätzlich die Perplexity-Schranke als Referenz.

Gewählt wird je Modell das k mit der höchsten Kohärenz unter allen k mit einer Diversität von mindestens 0,7.
Ergebnis: `results/chosen_k.json`, Verlauf: `results/figures/k_selection.png`.

## Installation und Ausführung

Voraussetzung: Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
python main.py
```

Für Schritt 1 die Datei `complaints.csv` von
<https://files.consumerfinance.gov/ccdb/complaints.csv.zip> entpacken und in den Ordner
`../complaints/` neben dem Repository legen. Liegt die Stichprobe schon vor, wird Schritt 1 übersprungen.

Laufzeit auf einem üblichen Laptop: Vorverarbeitung ca. 5 Minuten, Suche nach k ca. 30 bis 60 Minuten.

## Ergebnisse

Alle Tabellen liegen in `results/tables/`, alle Abbildungen in `results/figures/`.
