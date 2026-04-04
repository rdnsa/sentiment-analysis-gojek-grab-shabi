from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("playstore_reviews_processed_sentiment.csv")
OUTPUT_DIR = Path("market_insight_charts")

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {
    "positive": "#16a34a",
    "neutral": "#6b7280",
    "negative": "#dc2626",
}
BRAND_ORDER = ["gojek", "grab"]
BRAND_COLORS = {
    "gojek": "#00AA13",
    "grab": "#00B14F",
}
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

ISSUE_KEYWORDS = [
    (
        "Driver Availability",
        [
            "susah dapet driver",
            "susah dapat driver",
            "ga dapet driver",
            "gak dapet driver",
            "tidak ada driver",
            "nggak ada driver",
            "ga ada driver",
            "gak ada driver",
            "lama dapet driver",
            "lama dapat driver",
            "nyari driver",
            "mencari driver",
        ],
    ),
    (
        "App Performance",
        [
            "lelet",
            "lemot",
            "lambat",
            "loading",
            "lag",
            "error",
            "crash",
            "gagal",
            "force close",
        ],
    ),
    (
        "Price and Promo",
        [
            "ongkir",
            "mahal",
            "promo",
            "diskon",
            "flash sale",
            "biaya",
            "harga",
        ],
    ),
    (
        "Payment and Refund",
        [
            "saldo",
            "gopay",
            "refund",
            "pembayaran",
            "bayar",
            "cashless",
            "top up",
        ],
    ),
    (
        "Service Quality",
        [
            "pelayanan",
            "sopan",
            "cancel",
            "dibatalin",
            "dibatalkan",
            "lama",
            "nunggu",
            "keluhan",
            "komplain",
        ],
    ),
]


def load_data(file_path):
    df = pd.read_csv(file_path)

    required_columns = {"date", "brand", "sentiment", "content", "clean_content"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["clean_content"] = df["clean_content"].fillna(df["content"]).astype(str)

    df = df.dropna(subset=["date"]).copy()
    df = df[df["brand"].isin(BRAND_ORDER)]
    df = df[df["sentiment"].isin(SENTIMENT_ORDER)]

    if df.empty:
        raise ValueError("No valid records found after basic filtering.")

    df["review_day"] = df["date"].dt.floor("D")
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time
    df["weekday"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    return df


def get_available_brands(df):
    present = set(df["brand"].unique())
    return [brand for brand in BRAND_ORDER if brand in present]


def plot_daily_sentiment_trend(df):
    brands = get_available_brands(df)
    daily = (
        df.groupby(["brand", "review_day", "sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    daily["percentage"] = (
        daily["count"]
        / daily.groupby(["brand", "review_day"])["count"].transform("sum")
        * 100
    )

    fig, axes = plt.subplots(1, len(brands), figsize=(7 * len(brands), 5), sharey=True)
    if len(brands) == 1:
        axes = [axes]

    for idx, brand in enumerate(brands):
        brand_daily = daily[daily["brand"] == brand]
        pivot = (
            brand_daily
            .pivot(index="review_day", columns="sentiment", values="percentage")
            .reindex(columns=SENTIMENT_ORDER)
            .fillna(0)
            .sort_index()
        )

        ax = axes[idx]
        for sentiment in SENTIMENT_ORDER:
            ax.plot(
                pivot.index,
                pivot[sentiment],
                marker="o",
                linewidth=2,
                color=SENTIMENT_COLORS[sentiment],
                label=sentiment.title(),
            )

        ax.set_title(f"{brand.title()} Sentiment Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percentage (%)")
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Daily Sentiment Trend by Brand", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = OUTPUT_DIR / "01_daily_sentiment_trend.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_weekly_net_sentiment_score(df):
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    scored = df.assign(sentiment_score=df["sentiment"].map(score_map).fillna(0))

    weekly = (
        scored.groupby(["brand", "week_start"], as_index=False)
        .agg(total_reviews=("sentiment_score", "size"), score_sum=("sentiment_score", "sum"))
    )
    weekly["nss"] = (weekly["score_sum"] / weekly["total_reviews"] * 100).round(2)

    brands = get_available_brands(df)
    fig, ax = plt.subplots(figsize=(10, 5))

    for brand in brands:
        brand_weekly = weekly[weekly["brand"] == brand].sort_values("week_start")
        ax.plot(
            brand_weekly["week_start"],
            brand_weekly["nss"],
            marker="o",
            linewidth=2,
            label=brand.title(),
            color=BRAND_COLORS.get(brand, "#334155"),
        )

    ax.axhline(0, color="#111827", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Weekly Net Sentiment Score (NSS)")
    ax.set_xlabel("Week Start")
    ax.set_ylabel("NSS (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "02_weekly_net_sentiment_score.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_share_of_voice(df):
    daily_volume = (
        df.groupby(["review_day", "brand"], as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    brands = get_available_brands(df)
    pivot = (
        daily_volume
        .pivot(index="review_day", columns="brand", values="review_count")
        .reindex(columns=brands)
        .fillna(0)
        .sort_index()
    )
    share_pct = (pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    pivot.plot(
        kind="area",
        stacked=True,
        ax=axes[0],
        alpha=0.85,
        color=[BRAND_COLORS.get(brand, "#64748b") for brand in brands],
    )
    axes[0].set_title("Daily Share of Voice (Review Volume)")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, title="Brand")

    for brand in brands:
        axes[1].plot(
            share_pct.index,
            share_pct[brand],
            marker="o",
            linewidth=2,
            label=brand.title(),
            color=BRAND_COLORS.get(brand, "#334155"),
        )

    axes[1].set_title("Daily Share of Voice (Percentage)")
    axes[1].set_ylabel("Share (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, title="Brand")
    axes[1].set_xlabel("Date")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_daily_share_of_voice.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_negative_heatmap(df):
    negative_df = df[df["sentiment"] == "negative"].copy()
    if negative_df.empty:
        return None

    brands = get_available_brands(negative_df)
    fig, axes = plt.subplots(1, len(brands), figsize=(8 * len(brands), 5), sharey=True)
    if len(brands) == 1:
        axes = [axes]

    image = None
    for idx, brand in enumerate(brands):
        brand_df = negative_df[negative_df["brand"] == brand].copy()
        brand_df["weekday"] = pd.Categorical(brand_df["weekday"], categories=WEEKDAY_ORDER, ordered=True)

        heatmap_data = (
            brand_df
            .pivot_table(index="weekday", columns="hour", values="content", aggfunc="count", fill_value=0)
            .reindex(index=WEEKDAY_ORDER, columns=range(24), fill_value=0)
        )

        ax = axes[idx]
        image = ax.imshow(heatmap_data.values, aspect="auto", cmap="Reds")
        ax.set_title(f"{brand.title()} Negative Reviews")
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 2))
        ax.set_yticks(range(len(WEEKDAY_ORDER)))
        ax.set_yticklabels(WEEKDAY_ORDER)

    axes[0].set_ylabel("Weekday")
    fig.suptitle("Negative Sentiment Heatmap by Weekday and Hour", y=1.02)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="Review Count")

    fig.subplots_adjust(top=0.86, wspace=0.15, right=0.92)
    output_path = OUTPUT_DIR / "04_negative_heatmap_weekday_hour.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def categorize_issue(text):
    content = str(text).lower()
    for issue_name, keywords in ISSUE_KEYWORDS:
        if any(keyword in content for keyword in keywords):
            return issue_name
    return "Other"


def plot_negative_issue_gap_dumbbell(df):
    negative_df = df[df["sentiment"] == "negative"].copy()
    if negative_df.empty:
        return None

    negative_df["issue_category"] = negative_df["clean_content"].apply(categorize_issue)

    issue_counts = (
        negative_df.groupby(["brand", "issue_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    issue_counts["percentage"] = (
        issue_counts["count"]
        / issue_counts.groupby("brand")["count"].transform("sum")
        * 100
    )

    top_issues = (
        issue_counts.groupby("issue_category")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )

    plot_df = issue_counts[issue_counts["issue_category"].isin(top_issues)]
    brands = get_available_brands(df)
    if len(brands) < 2:
        return None

    pivot = (
        plot_df
        .pivot(index="issue_category", columns="brand", values="percentage")
        .reindex(columns=brands)
        .fillna(0)
    )
    pivot = pivot.sort_values(by=brands[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = np.arange(len(pivot))

    for y_pos, (_, row) in enumerate(pivot.iterrows()):
        left = row[brands[0]]
        right = row[brands[1]]
        ax.plot([left, right], [y_pos, y_pos], color="#9ca3af", linewidth=2)

    ax.scatter(
        pivot[brands[0]],
        y_positions,
        s=90,
        color=BRAND_COLORS.get(brands[0], "#334155"),
        label=brands[0].title(),
        zorder=3,
    )
    ax.scatter(
        pivot[brands[1]],
        y_positions,
        s=90,
        color=BRAND_COLORS.get(brands[1], "#1d4ed8"),
        label=brands[1].title(),
        zorder=3,
    )

    for y_pos, (_, row) in enumerate(pivot.iterrows()):
        ax.text(row[brands[0]] + 0.3, y_pos + 0.1, f"{row[brands[0]]:.1f}%", fontsize=8)
        ax.text(row[brands[1]] + 0.3, y_pos - 0.25, f"{row[brands[1]]:.1f}%", fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Share of Negative Reviews (%)")
    ax.set_title("Negative Issue Gap: Gojek vs Grab")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "05_negative_issue_gap_dumbbell.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_market_snapshot(df):
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    scored = df.assign(sentiment_score=df["sentiment"].map(score_map).fillna(0))

    latest_day = scored["review_day"].max()
    recent = scored[scored["review_day"] == latest_day].copy()

    day_summary = (
        recent.groupby("brand", as_index=False)
        .agg(
            total_reviews=("sentiment_score", "size"),
            positive=("sentiment", lambda s: (s == "positive").sum()),
            neutral=("sentiment", lambda s: (s == "neutral").sum()),
            negative=("sentiment", lambda s: (s == "negative").sum()),
            nss=("sentiment_score", lambda s: round((s.sum() / len(s)) * 100, 2)),
        )
        .sort_values("brand")
    )

    output_path = OUTPUT_DIR / "market_insight_snapshot_latest_day.csv"
    day_summary.to_csv(output_path, index=False)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT_PATH)

    outputs = [
        plot_daily_sentiment_trend(df),
        plot_weekly_net_sentiment_score(df),
        plot_share_of_voice(df),
        plot_negative_heatmap(df),
        plot_negative_issue_gap_dumbbell(df),
        create_market_snapshot(df),
    ]

    generated = [path for path in outputs if path is not None]
    print("Generated market insight files:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
