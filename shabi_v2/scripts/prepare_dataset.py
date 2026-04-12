from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

MIN_REQUIRED_SAMPLES = 3000


def clean_text(text: object) -> str:
    value = str(text or "").lower()
    value = URL_PATTERN.sub(" ", value)
    value = MENTION_PATTERN.sub(" ", value)
    value = HASHTAG_PATTERN.sub(" ", value)
    value = NON_ALNUM_PATTERN.sub(" ", value)
    value = WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def score_to_sentiment(score: int) -> str:
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def sentiment_to_score(sentiment: object) -> int:
    value = str(sentiment or "").strip().lower()
    if value == "negative":
        return 2
    if value == "neutral":
        return 3
    if value == "positive":
        return 5
    return 3


def find_latest_raw_csv(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("playstore_reviews_raw_*.csv"))
    latest_file = raw_dir / "playstore_reviews_raw_latest.csv"

    if latest_file.exists():
        return latest_file
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No raw file found in {raw_dir}. Run scrape_playstore_reviews.py first."
    )


def prepare_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_columns = {
        "date",
        "username",
        "content",
        "brand",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if "source" not in df.columns:
        df["source"] = "google_play"

    if "app_id" not in df.columns:
        app_map = {
            "gojek": "com.gojek.app",
            "grab": "com.grabtaxi.passenger",
        }
        df["app_id"] = df["brand"].astype(str).str.lower().map(app_map).fillna("unknown_app")

    if "score" not in df.columns:
        if "sentiment" in df.columns:
            df["score"] = df["sentiment"].apply(sentiment_to_score)
        else:
            df["score"] = 3

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].fillna(pd.Timestamp("1970-01-01"))
    df = df.copy()

    df["username"] = df["username"].fillna("unknown_user").astype(str).str.strip()
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df.loc[df["content"] == "", "content"] = "tidak ada komentar"
    df["clean_content"] = df["content"].apply(clean_text)
    df.loc[df["clean_content"] == "", "clean_content"] = "tidak ada komentar"

    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(3).astype(int)
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
        invalid_mask = ~df["sentiment"].isin({"negative", "neutral", "positive"})
        df.loc[invalid_mask, "sentiment"] = df.loc[invalid_mask, "score"].apply(score_to_sentiment)
    else:
        df["sentiment"] = df["score"].apply(score_to_sentiment)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["brand", "username", "clean_content", "score"]).reset_index(drop=True)
    print(f"[INFO] Removed duplicates: {before_dedup - len(df)}")

    ordered_columns = [
        "date",
        "username",
        "content",
        "clean_content",
        "score",
        "sentiment",
        "brand",
        "app_id",
        "source",
    ]
    return df[ordered_columns]


def save_outputs(df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = output_dir / f"playstore_reviews_processed_{timestamp}.csv"
    latest_path = output_dir / "playstore_reviews_processed_latest.csv"

    df.to_csv(timestamped_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"[INFO] Saved timestamped processed file: {timestamped_path}")
    print(f"[INFO] Saved latest processed file: {latest_path}")
    return timestamped_path, latest_path


def print_summary(df: pd.DataFrame) -> None:
    total_rows = len(df)
    class_dist = df["sentiment"].value_counts().sort_index()
    brand_dist = df["brand"].value_counts().sort_index()

    print("\n[CHECK] Prepared dataset summary")
    print(f"- Total rows: {total_rows}")
    print(f"- Meets minimum 3,000 rows: {'YES' if total_rows >= MIN_REQUIRED_SAMPLES else 'NO'}")

    print("\nClass distribution:")
    for label, count in class_dist.items():
        pct = (count / total_rows) * 100 if total_rows else 0
        print(f"- {label}: {count} ({pct:.2f}%)")

    print("\nBrand distribution:")
    for label, count in brand_dist.items():
        pct = (count / total_rows) * 100 if total_rows else 0
        print(f"- {label}: {count} ({pct:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and label scraped Play Store reviews for sentiment model training."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="Optional raw input CSV path. If omitted, the latest file in data/raw is used.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Raw data directory when input path is not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Processed output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path) if args.input_path else find_latest_raw_csv(Path(args.raw_dir))

    print(f"[INFO] Input raw CSV: {input_path}")
    prepared_df = prepare_data(input_path)
    save_outputs(prepared_df, Path(args.output_dir))
    print_summary(prepared_df)


if __name__ == "__main__":
    main()
