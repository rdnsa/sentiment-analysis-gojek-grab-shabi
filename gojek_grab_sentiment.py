import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from google_play_scraper import Sort, reviews
from transformers import AutoTokenizer, pipeline


# Google Play app IDs and output configuration
GOJEK_APP_ID = "com.gojek.app"
GRAB_APP_ID = "com.grabtaxi.passenger"
REVIEWS_PER_BRAND = 1500

RAW_DATA_PATH = Path("playstore_reviews_raw_gojek_grab.csv")
PROCESSED_DATA_PATH = Path("playstore_reviews_processed_sentiment.csv")
CHART_PATH = Path("playstore_sentiment_distribution_gojek_vs_grab.png")


# Regex patterns for text cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]+\b")


# Basic Indonesian stopwords for bonus word frequency analysis
STOPWORDS = {
    "dan",
    "yang",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "ini",
    "itu",
    "ada",
    "jadi",
    "karena",
    "saja",
    "aku",
    "kamu",
    "kita",
    "mereka",
    "kami",
    "gue",
    "aja",
    "nya",
    "ga",
    "gak",
    "nggak",
    "bgt",
    "udah",
    "sudah",
    "belum",
    "lebih",
    "masih",
    "juga",
    "atau",
    "the",
    "a",
    "an",
    "of",
    "to",
    "is",
    "are",
    "in",
    "on",
}


# Sentiment model candidates: prefer Indonesian, fallback to multilingual, then generic default
MODEL_CANDIDATES = [
    "w11wo/indonesian-roberta-base-sentiment-classifier",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "distilbert-base-uncased-finetuned-sst-2-english",
]


# Runtime objects initialized in main()
SENTIMENT_PIPELINE = None
MODEL_MAX_LENGTH = 512
MODEL_NUM_LABELS = 3


def scrape_data(query, limit):
    """Scrape Google Play reviews and return selected fields as a DataFrame."""
    rows = []
    seen = set()
    continuation_token = None

    while len(rows) < limit:
        count_to_fetch = min(200, limit - len(rows))
        batch, continuation_token = reviews(
            query,
            lang="id",
            country="id",
            sort=Sort.NEWEST,
            count=count_to_fetch,
            continuation_token=continuation_token,
            filter_score_with=None,
        )

        if not batch:
            break

        for item in batch:
            content = (item.get("content") or "").strip()
            if not content:
                continue

            row = {
                "date": item.get("at"),
                "username": item.get("userName") or "unknown_user",
                "content": content,
            }

            key = (row["username"], row["content"], str(row["date"]))
            if key in seen:
                continue

            seen.add(key)
            rows.append(row)

            if len(rows) >= limit:
                break

        if continuation_token is None:
            break

    # Top-up from MOST_RELEVANT if NEWEST cannot reach the target count
    if len(rows) < limit:
        count_to_fetch = min(200, limit - len(rows))
        fallback_batch, _ = reviews(
            query,
            lang="id",
            country="id",
            sort=Sort.MOST_RELEVANT,
            count=count_to_fetch,
            filter_score_with=None,
        )

        for item in fallback_batch:
            content = (item.get("content") or "").strip()
            if not content:
                continue

            row = {
                "date": item.get("at"),
                "username": item.get("userName") or "unknown_user",
                "content": content,
            }

            key = (row["username"], row["content"], str(row["date"]))
            if key in seen:
                continue

            seen.add(key)
            rows.append(row)

            if len(rows) >= limit:
                break

    return pd.DataFrame(rows, columns=["date", "username", "content"])


def clean_text(text):
    """Lowercase text and remove URLs, mentions, hashtags, and extra spaces."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def _load_sentiment_pipeline():
    """Load an Indonesian-first sentiment model with robust fallback behavior."""
    for model_name in MODEL_CANDIDATES:
        try:
            sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            model_max_length = getattr(tokenizer, "model_max_length", 512)
            if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 4096:
                model_max_length = 512

            model_num_labels = int(getattr(sentiment_pipe.model.config, "num_labels", 3) or 3)

            print(f"Loaded sentiment model: {model_name}")
            return sentiment_pipe, model_max_length, model_num_labels
        except Exception as exc:
            print(f"Model load failed ({model_name}): {exc}")

    raise RuntimeError("No sentiment model could be loaded. Please check internet connection and dependencies.")


def _normalize_label(raw_label, score, num_labels):
    """Map model-specific labels into: positive, negative, neutral."""
    label_text = str(raw_label).strip().lower()

    # Direct semantic labels from model outputs
    if "positive" in label_text or label_text in {"pos"}:
        normalized = "positive"
    elif "negative" in label_text or label_text in {"neg"}:
        normalized = "negative"
    elif "neutral" in label_text or label_text in {"neu"}:
        normalized = "neutral"
    # Numeric labels used by many checkpoints (example: LABEL_0, LABEL_1, LABEL_2)
    elif label_text.startswith("label_"):
        try:
            label_index = int(label_text.split("_")[-1])
        except ValueError:
            label_index = -1

        if num_labels >= 3:
            if label_index == 0:
                normalized = "negative"
            elif label_index == 1:
                normalized = "neutral"
            elif label_index == 2:
                normalized = "positive"
            else:
                normalized = "neutral"
        elif num_labels == 2:
            normalized = "negative" if label_index == 0 else "positive"
        else:
            normalized = "neutral"
    else:
        normalized = "neutral"

    # For binary models, force low-confidence predictions into neutral
    if num_labels == 2 and float(score) < 0.60:
        return "neutral"

    return normalized


def analyze_sentiment(text):
    """Predict sentiment safely with truncation handling for model token limits."""
    if SENTIMENT_PIPELINE is None:
        raise RuntimeError("Sentiment pipeline is not initialized. Run main() to initialize it first.")

    if not isinstance(text, str) or not text.strip():
        return "neutral"

    result = SENTIMENT_PIPELINE(
        text,
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
    )[0]

    return _normalize_label(
        raw_label=result.get("label", "neutral"),
        score=result.get("score", 0.0),
        num_labels=MODEL_NUM_LABELS,
    )


def _calculate_summary(df):
    """Create sentiment count and percentage tables per brand."""
    summary = (
        df.groupby(["brand", "sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "total_count"})
    )

    summary["percentage"] = (
        summary["total_count"]
        / summary.groupby("brand")["total_count"].transform("sum")
        * 100
    ).round(2)

    sentiment_order = ["positive", "neutral", "negative"]
    summary["sentiment"] = pd.Categorical(summary["sentiment"], categories=sentiment_order, ordered=True)
    summary = summary.sort_values(["brand", "sentiment"]).reset_index(drop=True)
    return summary


def _plot_sentiment_distribution(summary):
    """Generate and save bar chart for sentiment percentage by brand."""
    sentiment_order = ["positive", "neutral", "negative"]
    pivot = summary.pivot(index="brand", columns="sentiment", values="percentage").fillna(0)
    pivot = pivot.reindex(columns=sentiment_order, fill_value=0)

    ax = pivot.plot(kind="bar")
    ax.set_title("Sentiment Distribution: Gojek vs Grab")
    ax.set_xlabel("Brand")
    ax.set_ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.show()


def _top_words(text_series, top_n=10):
    """Return top-N frequent words from a text series after lightweight filtering."""
    counter = Counter()

    for text in text_series.dropna():
        for token in TOKEN_PATTERN.findall(str(text).lower()):
            if len(token) < 3:
                continue
            if token in STOPWORDS:
                continue
            counter[token] += 1

    return counter.most_common(top_n)


def _print_insights(summary):
    """Print concise insight statements for brand sentiment comparison."""
    def get_pct(brand, sentiment):
        row = summary[(summary["brand"] == brand) & (summary["sentiment"] == sentiment)]
        if row.empty:
            return 0.0
        return float(row["percentage"].iloc[0])

    gojek_pos = get_pct("gojek", "positive")
    grab_pos = get_pct("grab", "positive")
    gojek_neg = get_pct("gojek", "negative")
    grab_neg = get_pct("grab", "negative")

    print("\nSummary insights:")
    if gojek_pos > grab_pos:
        print(f"- Gojek has higher positive sentiment ({gojek_pos:.2f}%) than Grab ({grab_pos:.2f}%).")
    elif grab_pos > gojek_pos:
        print(f"- Grab has higher positive sentiment ({grab_pos:.2f}%) than Gojek ({gojek_pos:.2f}%).")
    else:
        print(f"- Positive sentiment is equal for both brands ({gojek_pos:.2f}%).")

    if gojek_neg > grab_neg:
        print(f"- Gojek has higher negative sentiment ({gojek_neg:.2f}%) than Grab ({grab_neg:.2f}%).")
    elif grab_neg > gojek_neg:
        print(f"- Grab has higher negative sentiment ({grab_neg:.2f}%) than Gojek ({gojek_neg:.2f}%).")
    else:
        print(f"- Negative sentiment is equal for both brands ({gojek_neg:.2f}%).")


def _print_top_words_by_brand_and_sentiment(df):
    """Bonus: print top 10 frequent words by brand and sentiment."""
    print("\nTop 10 frequent words per sentiment per brand:")
    sentiment_order = ["positive", "neutral", "negative"]

    for brand in ["gojek", "grab"]:
        for sentiment in sentiment_order:
            subset = df[(df["brand"] == brand) & (df["sentiment"] == sentiment)]["clean_content"]
            top_words = _top_words(subset, top_n=10)
            if top_words:
                words_str = ", ".join([f"{word} ({count})" for word, count in top_words])
            else:
                words_str = "-"
            print(f"- {brand} | {sentiment}: {words_str}")


def main():
    """End-to-end workflow: scrape Play Store reviews, analyze sentiment, summarize, and visualize."""
    global SENTIMENT_PIPELINE, MODEL_MAX_LENGTH, MODEL_NUM_LABELS

    print("Starting data collection from Google Play reviews...")

    # 1) Data collection from Google Play
    gojek_df = scrape_data(GOJEK_APP_ID, REVIEWS_PER_BRAND)
    grab_df = scrape_data(GRAB_APP_ID, REVIEWS_PER_BRAND)

    if gojek_df.empty and grab_df.empty:
        raise RuntimeError("No reviews were scraped. Please retry later or check internet connection.")

    gojek_df["brand"] = "gojek"
    gojek_df["source"] = "google_play"
    grab_df["brand"] = "grab"
    grab_df["source"] = "google_play"

    print(f"Gojek reviews collected: {len(gojek_df)} / {REVIEWS_PER_BRAND}")
    print(f"Grab reviews collected : {len(grab_df)} / {REVIEWS_PER_BRAND}")

    # 2) Combine and save raw dataset
    df = pd.concat([gojek_df, grab_df], ignore_index=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw dataset saved to: {RAW_DATA_PATH}")

    # 3) Clean text and remove duplicates
    df["clean_content"] = df["content"].apply(clean_text)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["brand", "username", "clean_content"]).reset_index(drop=True)
    print(f"Removed duplicates: {before_dedup - len(df)}")

    # 4) Load sentiment model and infer sentiment labels
    SENTIMENT_PIPELINE, MODEL_MAX_LENGTH, MODEL_NUM_LABELS = _load_sentiment_pipeline()
    print("Running sentiment analysis...")
    df["sentiment"] = df["clean_content"].apply(analyze_sentiment)

    # 5) Save processed dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")

    # 6) Aggregate analysis
    summary = _calculate_summary(df)
    print("\nSentiment comparison table (count and percentage):")
    print(summary.to_string(index=False))

    # 7) Visualization
    _plot_sentiment_distribution(summary)
    print(f"Chart saved to: {CHART_PATH}")

    # 8) Console insights
    _print_insights(summary)

    # 9) Bonus analysis: top frequent words by sentiment and brand
    _print_top_words_by_brand_and_sentiment(df)


if __name__ == "__main__":
    main()