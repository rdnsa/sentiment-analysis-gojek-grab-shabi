from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

import gojek_grab_sentiment
import market_insight_visuals


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "playstore_reviews_raw_gojek_grab.csv"
PROCESSED_DATA_PATH = BASE_DIR / "playstore_reviews_processed_sentiment.csv"
IMAGE_FILES = [
    BASE_DIR / "playstore_sentiment_distribution_gojek_vs_grab.png",
    BASE_DIR / "market_insight_charts" / "01_daily_sentiment_trend.png",
    BASE_DIR / "market_insight_charts" / "02_weekly_net_sentiment_score.png",
    BASE_DIR / "market_insight_charts" / "03_daily_share_of_voice.png",
    BASE_DIR / "market_insight_charts" / "04_negative_heatmap_weekday_hour.png",
    BASE_DIR / "market_insight_charts" / "05_negative_issue_gap_dumbbell.png",
]
SNAPSHOT_PATH = BASE_DIR / "market_insight_charts" / "market_insight_snapshot_latest_day.csv"


st.set_page_config(
    page_title="Gojek vs Grab Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)


def run_pipeline():
    gojek_grab_sentiment.main(show_plot=False)
    market_insight_visuals.main()


def list_output_files():
    candidates = [RAW_DATA_PATH, PROCESSED_DATA_PATH, SNAPSHOT_PATH, *IMAGE_FILES]
    return [path for path in candidates if path.exists()]


def get_mime_type(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def build_zip_bundle(file_paths):
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zip_file:
        for path in file_paths:
            if not path.exists():
                continue
            try:
                archive_name = str(path.relative_to(BASE_DIR))
            except ValueError:
                archive_name = path.name
            zip_file.write(path, arcname=archive_name)

    buffer.seek(0)
    return buffer.getvalue()


def render_data_preview():
    st.subheader("Data Preview")

    if RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(RAW_DATA_PATH)
        st.markdown(f"Raw data rows: **{len(raw_df):,}**")
        st.dataframe(raw_df.head(20), use_container_width=True)
    else:
        st.info("Raw CSV belum tersedia. Jalankan pipeline dulu.")

    if PROCESSED_DATA_PATH.exists():
        processed_df = pd.read_csv(PROCESSED_DATA_PATH)
        st.markdown(f"Processed data rows: **{len(processed_df):,}**")

        summary = (
            processed_df.groupby(["brand", "sentiment"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["brand", "sentiment"])
        )
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Processed CSV belum tersedia. Jalankan pipeline dulu.")


def render_visual_preview():
    st.subheader("Preview Visualisasi")
    existing_images = [path for path in IMAGE_FILES if path.exists()]

    if not existing_images:
        st.info("Belum ada image visualisasi. Jalankan pipeline dulu.")
        return

    columns = st.columns(2)
    for idx, image_path in enumerate(existing_images):
        column = columns[idx % 2]
        with column:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)


def render_downloads():
    st.subheader("Download Hasil")
    output_files = list_output_files()

    if not output_files:
        st.info("Belum ada file output untuk di-download.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_bytes = build_zip_bundle(output_files)
    st.download_button(
        label="Download semua output (.zip)",
        data=zip_bytes,
        file_name=f"gojek_grab_outputs_{timestamp}.zip",
        mime="application/zip",
        use_container_width=True,
    )

    for file_path in output_files:
        with file_path.open("rb") as file_obj:
            st.download_button(
                label=f"Download {file_path.name}",
                data=file_obj.read(),
                file_name=file_path.name,
                mime=get_mime_type(file_path),
                key=f"download_{file_path.as_posix()}",
                use_container_width=True,
            )


st.title("Analisis Sentimen Gojek vs Grab")
st.caption("Web lokal untuk scraping review Google Play, analisis sentimen, visualisasi, dan download hasil.")

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

control_col, info_col = st.columns([2, 1])
with control_col:
    run_clicked = st.button(
        "Run scraping + sentiment + visualisasi",
        type="primary",
        use_container_width=True,
    )

with info_col:
    if st.session_state["last_run"]:
        st.info(f"Last run: {st.session_state['last_run']}")
    else:
        st.info("Belum ada run pada sesi ini.")

if run_clicked:
    with st.spinner("Pipeline sedang berjalan. Proses ini bisa memakan waktu beberapa menit..."):
        try:
            run_pipeline()
            st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("Pipeline selesai. File output sudah siap.")
        except Exception as exc:
            st.exception(exc)

render_downloads()
render_visual_preview()
render_data_preview()
