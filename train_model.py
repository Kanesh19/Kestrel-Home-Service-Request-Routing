from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

CANONICAL = [
    "Installations",
    "Repairs",
    "Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
]
RENAME = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}
ACTIVE_TEAMS = [
    "Billing",
    "Filters & Consumables",
    "Installs & Demo",
    "Product Advice",
    "Repairs",
    "Returns & Replacement",
    "Warranty Claims",
]


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    # Some legacy rows contain UTF-8 bytes decoded as Latin-1/Windows-1252.
    for enc in ("latin1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            if fixed != text:
                text = fixed
                break
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_model_text(frame: pd.DataFrame) -> pd.Series:
    text = frame["request_text"].map(clean_text)
    product = frame.get("product_family", "").fillna("").map(clean_text).str.replace(" ", "_", regex=False)
    warranty = frame.get("warranty_status", "").fillna("").map(clean_text).str.replace(" ", "_", regex=False)
    channel = frame.get("channel", "").fillna("").map(clean_text)
    source = frame.get("source", "").fillna("").map(clean_text)
    return (
        text
        + " product_" + product
        + " warranty_" + warranty
        + " channel_" + channel
        + " source_" + source
    ).str.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kestrel service-request routing model.")
    parser.add_argument("--train", default="data/train.csv", help="Path to labeled training CSV")
    parser.add_argument("--out", default="model.joblib", help="Output model artifact")
    args = parser.parse_args()

    train_path = Path(args.train)
    out_path = Path(args.out)
    df = pd.read_csv(train_path)
    if "team_label" not in df.columns or "request_text" not in df.columns:
        raise ValueError("Training CSV must contain request_text and team_label columns")

    # The policy says responsibilities did not change on 15 Jan 2026; only names changed.
    y = df["team_label"].map(lambda x: RENAME.get(x, x))
    unknown = sorted(set(y.dropna()) - set(ACTIVE_TEAMS))
    if unknown:
        raise ValueError(f"Unexpected team labels: {unknown}")

    model_text = make_model_text(df)
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.995,
        sublinear_tf=True, max_features=60_000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        sublinear_tf=True, max_features=100_000,
    )
    x_word = word.fit_transform(model_text)
    x_char = char.fit_transform(model_text)
    x = hstack([x_word, x_char]).tocsr()

    clf = LinearSVC(C=2.0)
    clf.fit(x, y)

    artifact = {
        "word_vectorizer": word,
        "char_vectorizer": char,
        "classifier": clf,
        "active_teams": ACTIVE_TEAMS,
        "rename_map": RENAME,
        "training_rows": len(df),
        "feature_recipe": "word TF-IDF (1-2) + char_wb TF-IDF (3-5) + product/warranty/channel/source tokens",
        "model": "LinearSVC(C=2.0)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path, compress=3)
    print(f"Saved {out_path} ({out_path.stat().st_size:,} bytes)")
    print("Canonical teams:", ", ".join(ACTIVE_TEAMS))


if __name__ == "__main__":
    main()
