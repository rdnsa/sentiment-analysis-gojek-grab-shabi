from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    value = str(text or "").lower()
    value = URL_PATTERN.sub(" ", value)
    value = MENTION_PATTERN.sub(" ", value)
    value = HASHTAG_PATTERN.sub(" ", value)
    value = NON_ALNUM_PATTERN.sub(" ", value)
    value = WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def load_best_model_path(report_dir: Path) -> Path:
    best_path = report_dir / "best_experiment.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"File {best_path} not found. Run train_experiments.py first."
        )

    best_info = json.loads(best_path.read_text(encoding="utf-8"))
    model_path = Path(best_info["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Best model file not found: {model_path}")

    return model_path


def predict_with_sklearn(model_path: Path, texts: List[str]) -> List[str]:
    model = joblib.load(model_path)
    cleaned = [clean_text(text) for text in texts]
    return model.predict(cleaned).tolist()


def predict_with_bilstm(model_path: Path, model_dir: Path, texts: List[str]) -> List[str]:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    tokenizer_path = model_dir / "bilstm_tokenizer.json"
    label_encoder_path = model_dir / "bilstm_label_encoder.joblib"

    if not tokenizer_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            "BiLSTM artifacts not found (tokenizer/label encoder). Re-run train_experiments.py."
        )

    model = tf.keras.models.load_model(model_path)
    tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
    label_encoder = joblib.load(label_encoder_path)

    cleaned = [clean_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=120, padding="post", truncating="post")

    probabilities = model.predict(padded, verbose=0)
    label_ids = np.argmax(probabilities, axis=1)
    return label_encoder.inverse_transform(label_ids).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment inference and output categorical labels (negative/neutral/positive)."
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Input text for prediction. Can be used multiple times.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Optional CSV with column 'text' for batch inference.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory containing best_experiment.json.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model artifacts.",
    )
    return parser.parse_args()


def collect_texts(args: argparse.Namespace) -> List[str]:
    texts = list(args.text)

    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        if "text" not in input_df.columns:
            raise ValueError("input CSV must contain a 'text' column.")
        texts.extend(input_df["text"].astype(str).tolist())

    if not texts:
        texts = [
            "Aplikasinya cepat dan drivernya ramah sekali.",
            "Kadang susah dapat driver di jam sibuk.",
            "Biasa saja, tidak terlalu bagus tapi tidak buruk.",
        ]

    return texts


def main() -> None:
    args = parse_args()
    texts = collect_texts(args)

    model_path = load_best_model_path(Path(args.report_dir))

    if model_path.suffix == ".joblib":
        predictions = predict_with_sklearn(model_path, texts)
    elif model_path.suffix == ".keras":
        predictions = predict_with_bilstm(model_path, Path(args.model_dir), texts)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

    print("Inference output (categorical labels):")
    for idx, (text, pred) in enumerate(zip(texts, predictions), start=1):
        print(f"{idx}. [{pred}] {text}")


if __name__ == "__main__":
    main()
