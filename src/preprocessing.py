"""Schritt 2: Rohtexte zu sauberen, lemmatisierten Token-Folgen aufbereiten."""
import re

import pandas as pd
import spacy
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phraser, Phrases
from nltk.corpus import stopwords

from config import CLEAN_FILE, SAMPLE_FILE, TABLES

DOMAIN_STOPWORDS = {
    "xxxx", "xx", "would", "also", "could", "get", "go", "say", "tell", "ask",
    "call", "make", "take", "know", "want", "time", "day", "month", "year",
    "receive", "state", "since", "even", "still", "back", "one", "two",
    "please", "thank", "regard", "dear", "sincerely",
}

PATTERNS = [
    (re.compile(r"\{\$?[\d,.]+\}"), " "),
    (re.compile(r"x{2,}(/x{2,})*", re.IGNORECASE), " "),
    (re.compile(r"https?://\S+|www\.\S+"), " "),
    (re.compile(r"\S+@\S+"), " "),
    (re.compile(r"[^a-zA-Z\s]"), " "),
    (re.compile(r"\s+"), " "),
]


def normalize(text: str) -> str:
    """Entfernt Platzhalter, Beträge, Links, Zahlen und Satzzeichen."""
    text = text.lower()
    for pattern, repl in PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


def lemmatize(texts: list[str], stop: set[str]) -> list[list[str]]:
    """Tokenisiert und lemmatisiert mit spaCy und filtert Stoppwörter sowie kurze Tokens."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    result = []
    for doc in nlp.pipe(texts, batch_size=500):
        result.append([
            t.lemma_ for t in doc
            if t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
            and len(t.lemma_) > 2
            and t.lemma_ not in stop
        ])
    return result


def main() -> None:
    """Liest die Stichprobe, bereinigt sie und bildet Bigramme."""
    df = pd.read_csv(SAMPLE_FILE)
    stop = set(stopwords.words("english")) | DOMAIN_STOPWORDS

    tokens = lemmatize([normalize(t) for t in df["narrative"]], stop)

    phrases = Phrases(tokens, min_count=20, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS)
    bigram = Phraser(phrases)
    tokens = [bigram[doc] for doc in tokens]

    df["clean_text"] = [" ".join(doc) for doc in tokens]
    df["n_tokens"] = [len(doc) for doc in tokens]
    df = df[df["n_tokens"] >= 5].reset_index(drop=True)
    df.to_parquet(CLEAN_FILE, index=False)

    example = df.loc[0, ["narrative", "clean_text"]]
    (TABLES / "preprocessing_example.txt").write_text(
        f"ROH:\n{example['narrative'][:800]}\n\nBEREINIGT:\n{example['clean_text'][:600]}\n",
        encoding="utf-8",
    )
    print(f"{len(df)} Texte, Median {int(df['n_tokens'].median())} Tokens je Text")
    print(example["clean_text"][:400])


if __name__ == "__main__":
    main()
