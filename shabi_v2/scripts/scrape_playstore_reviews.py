from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from google_play_scraper import Sort, reviews


APP_CONFIG = [
    {"brand": "gojek", "app_id": "com.gojek.app"},
    {"brand": "grab", "app_id": "com.grabtaxi.passenger"},
]

MIN_REQUIRED_SAMPLES = 3000
HIGH_SCORE_TARGET_SAMPLES = 10000


def fetch_reviews_for_sort(
    app_id: str,
    brand: str,
    target_count: int,
    sort_mode: Sort,
    lang: str,
    country: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen_keys = set()
    continuation_token: Optional[str] = None

    while len(rows) < target_count:
        request_count = min(200, target_count - len(rows))

        batch, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort_mode,
            count=request_count,
            continuation_token=continuation_token,
            filter_score_with=None,
        )

        if not batch:
            break

        for item in batch:
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            row = {
                "date": item.get("at"),
                "username": str(item.get("userName") or "unknown_user").strip(),
                "content": content,
                "score": int(item.get("score") or 0),
                "thumbs_up_count": int(item.get("thumbsUpCount") or 0),
                "review_created_version": item.get("reviewCreatedVersion"),
                "brand": brand,
                "app_id": app_id,
                "source": "google_play",
            }

            dedupe_key = (row["username"], row["content"], str(row["date"]))
            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            rows.append(row)

            if len(rows) >= target_count:
                break

        if continuation_token is None:
            break

        # Keep request cadence polite to avoid temporary throttling.
        time.sleep(0.15)

    return rows


def fetch_reviews_for_app(
    app_id: str,
    brand: str,
    per_app_target: int,
    lang: str,
    country: str,
) -> List[Dict[str, object]]:
    newest_rows = fetch_reviews_for_sort(
        app_id=app_id,
        brand=brand,
        target_count=per_app_target,
        sort_mode=Sort.NEWEST,
        lang=lang,
        country=country,
    )

    if len(newest_rows) >= per_app_target:
        return newest_rows

    fallback_needed = per_app_target - len(newest_rows)
    fallback_rows = fetch_reviews_for_sort(
        app_id=app_id,
        brand=brand,
        target_count=fallback_needed,
        sort_mode=Sort.MOST_RELEVANT,
        lang=lang,
        country=country,
    )

    combined = newest_rows + fallback_rows
    deduped: List[Dict[str, object]] = []
    seen_keys = set()
    for row in combined:
        dedupe_key = (row["username"], row["content"], str(row["date"]))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(row)

    return deduped[:per_app_target]


def scrape_dataset(
    per_app_target: int,
    lang: str,
    country: str,
) -> pd.DataFrame:
    all_rows: List[Dict[str, object]] = []

    print("[INFO] Starting independent scraping from Google Play...")
    for app in APP_CONFIG:
        brand = app["brand"]
        app_id = app["app_id"]

        print(f"[INFO] Scraping {brand} ({app_id}) with target {per_app_target} rows...")
        try:
            app_rows = fetch_reviews_for_app(
                app_id=app_id,
                brand=brand,
                per_app_target=per_app_target,
                lang=lang,
                country=country,
            )
            all_rows.extend(app_rows)
            print(f"[INFO] {brand}: collected {len(app_rows)} rows")
        except Exception as exc:
            print(f"[WARN] Failed scraping {brand} ({app_id}): {exc}")

    if not all_rows:
        raise RuntimeError("No rows were scraped. Check internet access and app IDs.")

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    before_drop = len(df)
    df = df.drop_duplicates(subset=["brand", "username", "content", "date"]).reset_index(drop=True)
    dropped = before_drop - len(df)

    print(f"[INFO] Deduplicated rows removed: {dropped}")
    print(f"[INFO] Final scraped rows: {len(df)}")
    return df


def save_outputs(df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = output_dir / f"playstore_reviews_raw_{timestamp}.csv"
    latest_path = output_dir / "playstore_reviews_raw_latest.csv"

    df.to_csv(timestamped_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"[INFO] Saved timestamped file: {timestamped_path}")
    print(f"[INFO] Saved latest file: {latest_path}")
    return timestamped_path, latest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Google Play reviews independently for sentiment analysis submission."
    )
    parser.add_argument(
        "--per-app-target",
        type=int,
        default=5000,
        help="Target number of reviews per app (default: 5000 -> total target 10000).",
    )
    parser.add_argument("--lang", type=str, default="id", help="Language code for scraping (default: id).")
    parser.add_argument("--country", type=str, default="id", help="Country code for scraping (default: id).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory for raw CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = scrape_dataset(
        per_app_target=args.per_app_target,
        lang=args.lang,
        country=args.country,
    )
    save_outputs(df, Path(args.output_dir))

    total_rows = len(df)
    print("\n[CHECK] Submission readiness summary")
    print(f"- Total rows scraped: {total_rows}")
    print(f"- Meets minimum 3,000 rows: {'YES' if total_rows >= MIN_REQUIRED_SAMPLES else 'NO'}")
    print(f"- Meets optional 10,000 rows: {'YES' if total_rows >= HIGH_SCORE_TARGET_SAMPLES else 'NO'}")


if __name__ == "__main__":
    main()
