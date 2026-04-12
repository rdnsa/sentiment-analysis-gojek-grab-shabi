from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


MIN_TEST_ACCURACY = 0.85
TARGET_HIGH_ACCURACY = 0.92
LABEL_ORDER = ["negative", "neutral", "positive"]


@dataclass
class ExperimentResult:
    experiment_name: str
    algorithm: str
    feature_extraction: str
    split: str
    train_accuracy: float
    test_accuracy: float
    model_path: str
    notes: str


def find_latest_processed_csv(processed_dir: Path) -> Path:
    latest_file = processed_dir / "playstore_reviews_processed_latest.csv"
    if latest_file.exists():
        return latest_file

    candidates = sorted(processed_dir.glob("playstore_reviews_processed_*.csv"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No processed dataset found in {processed_dir}. Run prepare_dataset.py first."
    )


def load_training_data(input_path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(input_path)

    required_columns = {"clean_content", "sentiment"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["clean_content", "sentiment"]).copy()
    df["clean_content"] = df["clean_content"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(LABEL_ORDER)]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning filters.")

    return df["clean_content"], df["sentiment"]


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_experiment(
    experiment_name: str,
    algorithm: str,
    feature_extraction: str,
    split: str,
    y_train_true: pd.Series,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series,
    y_test_pred: np.ndarray,
    model_path: Path,
) -> Tuple[ExperimentResult, Dict[str, object]]:
    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)

    note_parts = []
    note_parts.append("PASS" if test_acc >= MIN_TEST_ACCURACY else "FAIL")
    if train_acc >= TARGET_HIGH_ACCURACY and test_acc >= TARGET_HIGH_ACCURACY:
        note_parts.append("HIGH_SCORE_READY")

    result = ExperimentResult(
        experiment_name=experiment_name,
        algorithm=algorithm,
        feature_extraction=feature_extraction,
        split=split,
        train_accuracy=float(train_acc),
        test_accuracy=float(test_acc),
        model_path=str(model_path.as_posix()),
        notes=" | ".join(note_parts),
    )

    report_dict = classification_report(
        y_test_true,
        y_test_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    payload = {
        "summary": result.__dict__,
        "classification_report": report_dict,
    }
    return result, payload


def run_experiment_svm_tfidf(
    x: pd.Series,
    y: pd.Series,
    model_dir: Path,
    figure_dir: Path,
) -> Tuple[ExperimentResult, Dict[str, object]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=60000,
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(C=1.0)),
        ]
    )

    pipeline.fit(x_train, y_train)
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    model_path = model_dir / "svm_tfidf_80_20.joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        labels=LABEL_ORDER,
        title="Experiment 1 - SVM + TF-IDF (80/20)",
        output_path=figure_dir / "cm_experiment_1_svm_tfidf.png",
    )

    return evaluate_experiment(
        experiment_name="experiment_1",
        algorithm="SVM",
        feature_extraction="TF-IDF",
        split="80/20",
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        model_path=model_path,
    )


def run_experiment_logreg_tfidf(
    x: pd.Series,
    y: pd.Series,
    model_dir: Path,
    figure_dir: Path,
) -> Tuple[ExperimentResult, Dict[str, object]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=80000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    model_path = model_dir / "logreg_tfidf_70_30.joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        labels=LABEL_ORDER,
        title="Experiment 2 - Logistic Regression + TF-IDF (70/30)",
        output_path=figure_dir / "cm_experiment_2_logreg_tfidf.png",
    )

    return evaluate_experiment(
        experiment_name="experiment_2",
        algorithm="Logistic Regression",
        feature_extraction="TF-IDF",
        split="70/30",
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        model_path=model_path,
    )


def run_experiment_bilstm(
    x: pd.Series,
    y: pd.Series,
    model_dir: Path,
    figure_dir: Path,
) -> Tuple[ExperimentResult, Dict[str, object]]:
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, LSTM
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.utils import to_categorical
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for experiment 3 (BiLSTM). Install it from requirements.txt."
        ) from exc

    from sklearn.preprocessing import LabelEncoder

    tf.random.set_seed(42)
    np.random.seed(42)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    max_words = 30000
    max_len = 120

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len, padding="post", truncating="post")
    x_test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_len, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_cat = to_categorical(y_train_encoded)
    y_test_cat = to_categorical(y_test_encoded)

    model = Sequential(
        [
            Embedding(input_dim=max_words, output_dim=128),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPool1D(),
            Dropout(0.30),
            Dense(64, activation="relu"),
            Dropout(0.20),
            Dense(len(label_encoder.classes_), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=2,
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train_seq,
        y_train_cat,
        validation_split=0.1,
        epochs=8,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    train_prob = model.predict(x_train_seq, verbose=0)
    test_prob = model.predict(x_test_seq, verbose=0)

    y_train_pred = label_encoder.inverse_transform(np.argmax(train_prob, axis=1))
    y_test_pred = label_encoder.inverse_transform(np.argmax(test_prob, axis=1))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "bilstm_sequence_80_20.keras"
    tokenizer_path = model_dir / "bilstm_tokenizer.json"
    label_encoder_path = model_dir / "bilstm_label_encoder.joblib"

    model.save(model_path)
    tokenizer_path.write_text(tokenizer.to_json(), encoding="utf-8")
    joblib.dump(label_encoder, label_encoder_path)

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        labels=LABEL_ORDER,
        title="Experiment 3 - BiLSTM + Sequence (80/20)",
        output_path=figure_dir / "cm_experiment_3_bilstm.png",
    )

    result, payload = evaluate_experiment(
        experiment_name="experiment_3",
        algorithm="BiLSTM",
        feature_extraction="Text Sequence Tokenizer",
        split="80/20",
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        model_path=model_path,
    )

    payload["artifacts"] = {
        "tokenizer_path": tokenizer_path.as_posix(),
        "label_encoder_path": label_encoder_path.as_posix(),
    }
    return result, payload


def save_reports(
    results: List[ExperimentResult],
    detailed_reports: Dict[str, Dict[str, object]],
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([item.__dict__ for item in results])
    summary_df = summary_df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)

    metrics_csv = report_dir / "metrics_experiments.csv"
    metrics_json = report_dir / "metrics_experiments.json"
    best_json = report_dir / "best_experiment.json"

    summary_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(
        json.dumps({"experiments": detailed_reports}, indent=2),
        encoding="utf-8",
    )

    best_record = summary_df.iloc[0].to_dict()
    best_json.write_text(json.dumps(best_record, indent=2), encoding="utf-8")

    print("\n[INFO] Experiment summary")
    print(summary_df.to_string(index=False))
    print(f"\n[INFO] Saved metrics CSV : {metrics_csv}")
    print(f"[INFO] Saved metrics JSON: {metrics_json}")
    print(f"[INFO] Saved best model info: {best_json}")

    passed_85 = int((summary_df["test_accuracy"] >= MIN_TEST_ACCURACY).sum())
    passed_92 = int(
        ((summary_df["test_accuracy"] >= TARGET_HIGH_ACCURACY) & (summary_df["train_accuracy"] >= TARGET_HIGH_ACCURACY)).sum()
    )

    print("\n[CHECK] Submission target validation")
    print(f"- Experiments with test accuracy >= 85%: {passed_85}")
    print(f"- Experiments with train & test >= 92%: {passed_92}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three sentiment-training experiment schemes for submission readiness."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="Optional processed CSV path. Defaults to latest file in data/processed.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed directory for auto-discovery.",
    )
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to store model artifacts.")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory to store metrics and logs.")
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="reports/figures",
        help="Directory for confusion matrix plots.",
    )
    parser.add_argument(
        "--skip-deep-learning",
        action="store_true",
        help="Skip experiment 3 (BiLSTM) if TensorFlow is not available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path) if args.input_path else find_latest_processed_csv(Path(args.processed_dir))
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    figure_dir = Path(args.figure_dir)

    print(f"[INFO] Using processed dataset: {input_path}")
    x, y = load_training_data(input_path)

    results: List[ExperimentResult] = []
    detailed_reports: Dict[str, Dict[str, object]] = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "dataset_path": input_path.as_posix(),
            "num_samples": int(len(x)),
        }
    }

    exp1_result, exp1_payload = run_experiment_svm_tfidf(x, y, model_dir, figure_dir)
    results.append(exp1_result)
    detailed_reports[exp1_result.experiment_name] = exp1_payload

    exp2_result, exp2_payload = run_experiment_logreg_tfidf(x, y, model_dir, figure_dir)
    results.append(exp2_result)
    detailed_reports[exp2_result.experiment_name] = exp2_payload

    if not args.skip_deep_learning:
        try:
            exp3_result, exp3_payload = run_experiment_bilstm(x, y, model_dir, figure_dir)
            results.append(exp3_result)
            detailed_reports[exp3_result.experiment_name] = exp3_payload
        except Exception as exc:
            print(f"[WARN] Deep learning experiment failed: {exc}")
            print("[WARN] Re-run after installing TensorFlow or use --skip-deep-learning.")
    else:
        print("[INFO] Deep learning experiment skipped by argument.")

    if not results:
        raise RuntimeError("No experiment was completed.")

    save_reports(results, detailed_reports, report_dir)


if __name__ == "__main__":
    main()
