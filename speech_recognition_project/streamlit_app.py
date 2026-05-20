"""Streamlit UI for dataset ingestion, training, and inference."""

from __future__ import annotations

import json
import tempfile
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.datasets.common_voice import (
    ArchiveExtractionError,
    build_common_voice_metadata,
    discover_common_voice_root,
    extract_archive,
    find_archives,
)
from src.inference.predict import InferenceEngine
from src.models.ann_model import ANNModel
from src.models.svm_model import SVMModel
from src.models.train import train_pipeline
from src.utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
HARDWARE_ROOT = REPO_ROOT / "hardware"
if str(HARDWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(HARDWARE_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
HYBRID_ACCENT_METADATA_CSV = PROJECT_ROOT / "data" / "hybrid_accent_metadata.csv"
METADATA_CSV = (
    HYBRID_ACCENT_METADATA_CSV
    if HYBRID_ACCENT_METADATA_CSV.exists()
    else DEFAULT_METADATA_CSV
)
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTION_HISTORY_CSV = PROJECT_ROOT / "data" / "prediction_history.csv"
LATEST_TRAINING_REPORT_JSON = PROJECT_ROOT / "data" / "latest_accent_training_report.json"
LATEST_WORD_TRAINING_REPORT_JSON = PROJECT_ROOT / "data" / "latest_word_training_report.json"
ACCENT_GROUPS = ["coastal", "nairobi", "upcountry"]
DEFAULT_PI_MIC_DEVICE_INDEX = 1
PROMPT_WORDS_PATH = HARDWARE_ROOT / "prompt_words.txt"
METADATA_COLUMNS = [
    "file_path",
    "word_label",
    "accent_label",
    "speaker_id",
    "duration_sec",
    "split",
]


st.set_page_config(page_title="Kiswahili ASR", layout="wide")
st.title("Accent-Aware Kiswahili Speech Recognition")
st.caption(f"Active metadata: {METADATA_CSV}")


def _load_svm_engine(model_path: Path, scaler_path: Path, encoder_path: Path) -> InferenceEngine:
    model = SVMModel().load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return InferenceEngine(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        silence_threshold_db=55.0,
    )


def _artifact_exists(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def _accent_readiness(path: Path) -> tuple[bool, set[str]]:
    if not path.exists():
        return False, set()
    frame = pd.read_csv(path)
    accents = set(frame.get("accent_label", pd.Series(dtype=str)).dropna().astype(str))
    return set(ACCENT_GROUPS).issubset(accents), accents


def _word_prompts(metadata_path: Path, encoder_path: Path) -> list[str]:
    if PROMPT_WORDS_PATH.exists():
        words = [
            line.strip().lower()
            for line in PROMPT_WORDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if words:
            return words
    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
            return sorted(str(label) for label in encoder.classes_)
        except Exception:
            pass
    if metadata_path.exists():
        frame = pd.read_csv(metadata_path)
        if "word_label" in frame.columns:
            return sorted(frame["word_label"].dropna().astype(str).unique())
    return []


def _show_supported_words(words: list[str]) -> None:
    if not words:
        return
    st.caption("Supported spoken words for this prototype:")
    st.write(" · ".join(f"`{word}`" for word in words))


def _record_from_pi_mic(duration: float, device_index: int) -> str:
    from accent_hardware_runner import GpioController, record_audio
    import soundfile as sf

    gpio = None
    try:
        gpio = GpioController()
        gpio.led(True)
        gpio.beep(count=1, duration=0.08)
        audio = record_audio(duration=duration, device_index=device_index)
    finally:
        if gpio is not None:
            gpio.beep(count=2, duration=0.08)
            gpio.led(False)
            gpio.cleanup()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        return tmp.name


def _audio_level_summary(audio: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    rms_db = float(20 * np.log10(rms + 1e-9))
    return {
        "duration": len(audio) / sample_rate,
        "peak": peak,
        "rms_db": rms_db,
    }


def _show_audio_level(path: str) -> dict[str, float]:
    from src.utils.audio import load_audio

    audio, sample_rate = load_audio(path)
    summary = _audio_level_summary(audio, sample_rate)
    cols = st.columns(3)
    cols[0].metric("Captured seconds", f"{summary['duration']:.2f}")
    cols[1].metric("Peak level", f"{summary['peak']:.4f}")
    cols[2].metric("RMS dB", f"{summary['rms_db']:.1f}")
    if summary["peak"] < 0.01:
        st.warning(
            "The recording is extremely quiet. Check the mic mute switch, speak closer, "
            "or try increasing the recording time."
        )
    return summary


def _load_prediction_history() -> pd.DataFrame:
    if not PREDICTION_HISTORY_CSV.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "source",
                "prompt_word",
                "predicted_accent",
                "confidence_percent",
                "word_prediction",
                "word_confidence_percent",
                "audio_peak",
                "audio_rms_db",
                "top_predictions",
                "error",
                "audio_path",
            ]
        )
    return pd.read_csv(PREDICTION_HISTORY_CSV)


def _append_prediction_history(
    *,
    audio_path: str,
    source: str,
    prompt_word: str,
    audio_summary: dict[str, float],
    accent_result,
    word_result=None,
) -> None:
    history = _load_prediction_history()
    error = accent_result.error if accent_result and accent_result.is_error else ""
    top_predictions = ""
    if accent_result and not accent_result.is_error:
        top_predictions = "; ".join(
            f"{label}:{probability * 100:.1f}%"
            for label, probability in accent_result.top_k
        )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "prompt_word": prompt_word,
        "predicted_accent": "" if error else accent_result.predicted_word,
        "confidence_percent": "" if error else round(accent_result.confidence * 100, 2),
        "word_prediction": (
            word_result.predicted_word
            if word_result is not None and not word_result.is_error
            else ""
        ),
        "word_confidence_percent": (
            round(word_result.confidence * 100, 2)
            if word_result is not None and not word_result.is_error
            else ""
        ),
        "audio_peak": round(audio_summary.get("peak", 0.0), 5),
        "audio_rms_db": round(audio_summary.get("rms_db", -180.0), 2),
        "top_predictions": top_predictions,
        "error": error,
        "audio_path": audio_path,
    }
    updated = pd.concat([pd.DataFrame([row]), history], ignore_index=True).head(200)
    PREDICTION_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(PREDICTION_HISTORY_CSV, index=False)


def _render_prediction_history() -> None:
    st.subheader("Prediction History")
    history = _load_prediction_history()
    if history.empty:
        st.info("No predictions have been recorded yet.")
        return

    display_columns = [
        "timestamp",
        "source",
        "predicted_accent",
        "confidence_percent",
        "audio_peak",
        "audio_rms_db",
        "top_predictions",
        "error",
    ]
    available_columns = [column for column in display_columns if column in history.columns]
    st.dataframe(history[available_columns].head(20), use_container_width=True)
    if st.button("Clear Prediction History"):
        PREDICTION_HISTORY_CSV.unlink(missing_ok=True)
        st.rerun()


def _load_trainer_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=METADATA_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(METADATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame["accent_label"].isin(ACCENT_GROUPS)]
    return frame[METADATA_COLUMNS]


def _rebuild_hybrid_metadata() -> pd.DataFrame:
    frames = [
        _load_trainer_metadata(PROJECT_ROOT / "data" / "accent_metadata.csv"),
        _load_trainer_metadata(PROJECT_ROOT / "data" / "common_voice_accent_metadata.csv"),
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError("No accent metadata rows found. Record speakers or import data first.")
    frame = frame.drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    HYBRID_ACCENT_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(HYBRID_ACCENT_METADATA_CSV, index=False)
    return frame


def _save_training_report(report: dict, output_path: Path = LATEST_TRAINING_REPORT_JSON) -> None:
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(report["accuracy"]),
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "f1": float(report["f1"]),
        "wer": float(report["wer"]),
        "per_accent": {
            str(label): float(value)
            for label, value in report.get("per_accent", {}).items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _render_latest_training_report(
    report_path: Path = LATEST_TRAINING_REPORT_JSON,
    title: str = "Latest accent training",
) -> None:
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.caption(f"{title}: {report.get('timestamp', 'unknown')}")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{report.get('accuracy', 0) * 100:.2f}%")
    cols[1].metric("Precision", f"{report.get('precision', 0) * 100:.2f}%")
    cols[2].metric("Recall", f"{report.get('recall', 0) * 100:.2f}%")
    cols[3].metric("F1", f"{report.get('f1', 0) * 100:.2f}%")
    if report.get("per_accent"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"accent": accent, "accuracy_percent": accuracy * 100}
                    for accent, accuracy in report["per_accent"].items()
                ]
            ),
            use_container_width=True,
        )


def _run_demo_recognition(
    tmp_path: str,
    source: str,
    audio_summary: dict[str, float],
    selected_word: str,
    word_ready: bool,
    accent_ready: bool,
    word_model_path: Path,
    word_scaler_path: Path,
    word_encoder_path: Path,
    accent_model_path: Path,
    accent_scaler_path: Path,
    accent_encoder_path: Path,
) -> None:
    word_result = None
    if not word_ready:
        st.info("Word model artifacts are missing, so this run will classify accent only.")
    else:
        try:
            word_engine = _load_svm_engine(word_model_path, word_scaler_path, word_encoder_path)
            word_result = word_engine.predict_from_file(tmp_path)
        except Exception as exc:
            word_result = None
            st.error(f"Word recognition failed: {exc}")

        if word_result and word_result.is_error:
            st.error(word_result.error)
        elif word_result:
            is_match = (
                selected_word.strip().lower() == word_result.predicted_word.strip().lower()
                if selected_word
                else False
            )
            result_cols = st.columns(4)
            result_cols[0].metric("Prompt word", selected_word or "Not selected")
            result_cols[1].metric("Predicted spoken word", word_result.predicted_word)
            result_cols[2].metric("Word confidence", f"{word_result.confidence * 100:.1f}%")
            result_cols[3].metric("Prompt match", "Yes" if is_match else "No")
            st.caption("Word alternatives")
            word_probabilities = pd.DataFrame(
                word_result.top_k,
                columns=["word", "probability"],
            )
            word_probabilities["Confidence %"] = word_probabilities["probability"] * 100
            st.dataframe(
                word_probabilities[["word", "Confidence %"]],
                use_container_width=True,
            )

    st.divider()
    st.subheader("Accent Group")
    if accent_ready:
        accent_result = None
        try:
            accent_engine = _load_svm_engine(accent_model_path, accent_scaler_path, accent_encoder_path)
            accent_result = accent_engine.predict_from_file(tmp_path)
            if accent_result.is_error:
                st.error(accent_result.error)
            else:
                result_cols = st.columns(2)
                result_cols[0].metric("Speaker accent group", accent_result.predicted_word)
                result_cols[1].metric(
                    "Model confidence",
                    f"{accent_result.confidence * 100:.1f}%",
                )
                probabilities = pd.DataFrame(
                    accent_result.top_k,
                    columns=["accent_group", "probability"],
                )
                probabilities["percentage"] = probabilities["probability"] * 100
                st.bar_chart(probabilities.set_index("accent_group"))
                st.dataframe(
                    probabilities[["accent_group", "percentage"]].rename(
                        columns={"accent_group": "Accent", "percentage": "Confidence %"}
                    ),
                    use_container_width=True,
                )
            if accent_result is not None:
                _append_prediction_history(
                    audio_path=tmp_path,
                    source=source,
                    prompt_word=selected_word,
                    audio_summary=audio_summary,
                    accent_result=accent_result,
                    word_result=word_result,
                )
        except Exception as exc:
            st.error(f"Accent classification failed: {exc}")
    else:
        st.info(
            "No trained accent classifier is available yet. Add examples labelled "
            "`coastal`, `nairobi`, and `upcountry`, then train target `accent`."
        )


def _pipeline_steps() -> None:
    cols = st.columns(5)
    steps = [
        ("1", "Prompt word", "Choose the word the speaker must pronounce."),
        ("2", "Speaker audio", "Record or upload the speaker pronouncing that word."),
        ("3", "MFCC features", "Convert audio into a 39-value acoustic feature vector."),
        ("4", "Word text", "Check whether the model heard the target word."),
        ("5", "Accent group", "Classify speaker accent as Coastal, Nairobi, or Upcountry."),
    ]
    for col, (num, title, detail) in zip(cols, steps):
        with col:
            st.metric(num, title)
            st.caption(detail)


def _metadata_summary(path: Path) -> None:
    if not path.exists():
        st.info("No metadata CSV has been generated yet.")
        return
    frame = pd.read_csv(path)
    if frame.empty:
        st.warning("Metadata CSV exists but has no rows.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Labels", f"{frame['word_label'].nunique():,}")
    c3.metric("Accents", f"{frame['accent_label'].nunique():,}")
    c4.metric("Speakers", f"{frame['speaker_id'].nunique():,}")

    left, right = st.columns(2)
    with left:
        st.caption("Split counts")
        st.dataframe(frame["split"].value_counts().rename("rows"), use_container_width=True)
    with right:
        st.caption("Top labels")
        st.dataframe(frame["word_label"].value_counts().head(20).rename("rows"), use_container_width=True)

    low_counts = frame["word_label"].value_counts()
    if (low_counts < 2).any():
        st.warning(
            "Some labels have only one example. Training may fail or generalize poorly "
            "until the dataset has repeated examples per label."
        )


demo_tab, dataset_tab, train_tab = st.tabs(["Demo", "Dataset", "Train"])

with demo_tab:
    st.subheader("Speaker Audio to Text and Accent Group")
    _pipeline_steps()

    word_model_path = MODELS_DIR / "svm_model.joblib"
    word_scaler_path = MODELS_DIR / "scaler.joblib"
    word_encoder_path = MODELS_DIR / "label_encoder.joblib"
    accent_model_path = MODELS_DIR / "accent_svm_model.joblib"
    accent_scaler_path = MODELS_DIR / "accent_scaler.joblib"
    accent_encoder_path = MODELS_DIR / "accent_label_encoder.joblib"

    word_ready = _artifact_exists(word_model_path, word_scaler_path, word_encoder_path)
    accent_ready = _artifact_exists(accent_model_path, accent_scaler_path, accent_encoder_path)
    accent_data_ready, known_accents = _accent_readiness(METADATA_CSV)

    status_cols = st.columns(3)
    status_cols[0].metric("Word recognizer", "Ready" if word_ready else "Missing")
    status_cols[1].metric("Accent classifier", "Ready" if accent_ready else "Missing")
    status_cols[2].metric("Accent groups in data", f"{len(known_accents)}")

    if not accent_data_ready:
        st.warning(
            "Accent classification into Coastal, Nairobi, and Upcountry requires training data "
            "with those exact accent labels. The current metadata does not contain all three groups."
        )

    prompts = _word_prompts(METADATA_CSV, word_encoder_path)
    _show_supported_words(prompts)
    if prompts:
        selected_word = st.selectbox("Word for speaker to pronounce", prompts)
    else:
        selected_word = st.text_input("Word for speaker to pronounce", value="")

    st.markdown(f"### Pronounce: `{selected_word or 'choose a word'}`")
    st.caption("Ask the speaker to say this word once, clearly, then run recognition.")

    st.subheader("Raspberry Pi USB Microphone")
    mic_cols = st.columns(2)
    pi_mic_duration = mic_cols[0].number_input(
        "Recording seconds",
        min_value=1.0,
        max_value=10.0,
        value=2.5,
        step=0.5,
    )
    pi_mic_device_index = mic_cols[1].number_input(
        "Pi microphone device index",
        min_value=0,
        value=DEFAULT_PI_MIC_DEVICE_INDEX,
        step=1,
    )
    action_cols = st.columns(2)
    run_pi_mic = action_cols[0].button(
        "Record From Raspberry Pi USB Mic",
        type="primary",
        use_container_width=True,
    )
    if action_cols[1].button("Next Speaker / Clear Result", use_container_width=True):
        st.rerun()

    st.divider()
    st.subheader("Fallback Audio Input")
    uploaded_audio = st.file_uploader(
        "Upload speaker audio",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Use a short recording of the speaker pronouncing the prompt word.",
    )
    run_uploaded_audio = st.button("Run Uploaded Audio")

    tmp_path = None
    audio_summary = {}
    audio_source = ""
    if run_pi_mic:
        try:
            with st.spinner("Recording from the Raspberry Pi USB microphone..."):
                tmp_path = _record_from_pi_mic(
                    duration=float(pi_mic_duration),
                    device_index=int(pi_mic_device_index),
                )
            audio_source = "pi_usb_mic"
            st.audio(tmp_path)
            audio_summary = _show_audio_level(tmp_path)
        except Exception as exc:
            st.error(f"Raspberry Pi microphone recording failed: {exc}")
    elif run_uploaded_audio and uploaded_audio is not None:
        suffix = Path(getattr(uploaded_audio, "name", "recording.wav")).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_audio.getbuffer())
            tmp_path = tmp.name
        audio_source = "uploaded_audio"
        audio_summary = _show_audio_level(tmp_path)

    if tmp_path is not None:
        _run_demo_recognition(
            tmp_path=tmp_path,
            source=audio_source,
            audio_summary=audio_summary,
            selected_word=selected_word,
            word_ready=word_ready,
            accent_ready=accent_ready,
            word_model_path=word_model_path,
            word_scaler_path=word_scaler_path,
            word_encoder_path=word_encoder_path,
            accent_model_path=accent_model_path,
            accent_scaler_path=accent_scaler_path,
            accent_encoder_path=accent_encoder_path,
        )

    st.divider()
    _render_prediction_history()

with dataset_tab:
    st.subheader("Dataset Extraction")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archives = find_archives(RAW_DIR)
    archive_names = [archive.name for archive in archives]

    if archives:
        selected = st.selectbox("Archive", archive_names)
        archive_path = archives[archive_names.index(selected)]
        st.caption(str(archive_path))
        overwrite = st.checkbox("Overwrite existing extracted files", value=False)
        if st.button("Extract Archive", type="primary"):
            try:
                with st.spinner("Extracting dataset archive..."):
                    result = extract_archive(archive_path, RAW_DIR, overwrite=overwrite)
                st.success(f"Extracted {result.extracted_files:,} files.")
                if result.common_voice_root:
                    st.info(f"Common Voice root: {result.common_voice_root}")
            except ArchiveExtractionError as exc:
                st.error(str(exc))
    else:
        st.info("Place a Common Voice .tar, .tar.gz, .tgz, or .zip archive in data/raw.")

    st.divider()
    st.subheader("Metadata Generation")
    cv_root = discover_common_voice_root(RAW_DIR)
    cv_root_text = st.text_input("Common Voice root", value=str(cv_root or RAW_DIR))
    label_strategy = st.selectbox("Label strategy", ["sentence", "first_word"], index=0)
    single_word_only = st.checkbox("Keep single-word prompts only", value=False)
    min_label_count = st.number_input("Minimum examples per label", min_value=1, value=2, step=1)
    max_labels = st.number_input("Maximum labels", min_value=0, value=25, step=5)
    max_rows = st.number_input("Maximum rows", min_value=0, value=0, step=100)

    if st.button("Build Metadata CSV"):
        try:
            with st.spinner("Building metadata.csv..."):
                result = build_common_voice_metadata(
                    common_voice_root=cv_root_text,
                    output_csv=METADATA_CSV,
                    data_dir=RAW_DIR,
                    label_strategy=label_strategy,
                    max_rows=int(max_rows) or None,
                    single_word_only=single_word_only,
                    min_label_count=int(min_label_count),
                    max_labels=int(max_labels) or None,
                )
            st.success(
                f"Wrote {result.rows:,} rows, {result.labels:,} labels, "
                f"{result.accents:,} accent groups to {result.output_csv}."
            )
            if result.skipped_rows:
                st.warning(f"Skipped {result.skipped_rows:,} rows without usable audio or labels.")
        except Exception as exc:
            st.error(str(exc))

    _metadata_summary(METADATA_CSV)

with train_tab:
    st.subheader("Model Training")
    _metadata_summary(METADATA_CSV)
    _render_latest_training_report()
    _render_latest_training_report(
        report_path=LATEST_WORD_TRAINING_REPORT_JSON,
        title="Latest word recognition training",
    )
    st.divider()
    st.subheader("One-Click Accent Retraining")
    st.caption(
        "Rebuilds the hybrid metadata from local recordings and Common Voice support, "
        "then trains the accent SVM used by the live demo."
    )
    if st.button("Rebuild Metadata and Retrain Accent Model", type="primary"):
        try:
            with st.spinner("Rebuilding hybrid accent metadata..."):
                hybrid_frame = _rebuild_hybrid_metadata()
            st.success(f"Hybrid metadata rebuilt with {len(hybrid_frame):,} rows.")
            st.dataframe(
                hybrid_frame["accent_label"].value_counts().rename("rows"),
                use_container_width=True,
            )

            config = Config.load(str(PROJECT_ROOT / "config" / "default.yaml"))
            with st.spinner("Training accent SVM model... this can take a few minutes."):
                _, report = train_pipeline(
                    metadata_csv=str(HYBRID_ACCENT_METADATA_CSV),
                    data_dir=str(RAW_DIR),
                    model_type="svm",
                    config=config,
                    save_dir=str(MODELS_DIR),
                    target_column="accent_label",
                )
            _save_training_report(report)
            st.success("Accent model retrained and saved.")
            st.json(
                {
                    key: report[key]
                    for key in ("accuracy", "precision", "recall", "f1", "wer")
                }
            )
            st.caption("Per-accent accuracy")
            st.json(report["per_accent"])
            st.info("The next demo prediction will use the newly trained model.")
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")

    st.divider()
    st.subheader("One-Click Word Recognition Retraining")
    st.caption(
        "Trains the isolated-word recognizer from local USB microphone recordings. "
        "This is the speech-to-text component for the 20 prompt words."
    )
    if st.button("Retrain Spoken Word Model"):
        try:
            local_metadata = PROJECT_ROOT / "data" / "accent_metadata.csv"
            if not local_metadata.exists():
                raise FileNotFoundError(f"Missing local metadata: {local_metadata}")
            config = Config.load(str(PROJECT_ROOT / "config" / "default.yaml"))
            with st.spinner("Training spoken word SVM model... this can take a few minutes."):
                _, report = train_pipeline(
                    metadata_csv=str(local_metadata),
                    data_dir=str(RAW_DIR),
                    model_type="svm",
                    config=config,
                    save_dir=str(MODELS_DIR),
                    target_column="word_label",
                )
            _save_training_report(report, output_path=LATEST_WORD_TRAINING_REPORT_JSON)
            st.success("Spoken word model retrained and saved.")
            st.json(
                {
                    key: report[key]
                    for key in ("accuracy", "precision", "recall", "f1", "wer")
                }
            )
            st.info("The next demo prediction will show the newly trained spoken word result.")
        except Exception as exc:
            st.error(f"Word retraining failed: {exc}")

    st.divider()
    target = st.selectbox("Target", ["word", "accent"], index=0)
    if target == "accent":
        ready, accents = _accent_readiness(METADATA_CSV)
        if ready:
            st.success("Metadata contains Coastal, Nairobi, and Upcountry labels.")
        else:
            st.warning(
                "Accent training needs all three labels: coastal, nairobi, upcountry. "
                f"Current labels: {', '.join(sorted(accents)) if accents else 'none'}"
            )
    model_type = st.selectbox("Model", ["svm", "ann"], index=0)
    config_path = st.text_input("Config path", value=str(PROJECT_ROOT / "config" / "default.yaml"))
    save_dir = st.text_input("Model output directory", value=str(MODELS_DIR))

    if st.button("Train Model", type="primary"):
        try:
            config = Config.load(config_path)
            with st.spinner("Training model..."):
                _, report = train_pipeline(
                    metadata_csv=str(METADATA_CSV),
                    data_dir=str(RAW_DIR),
                    model_type=model_type,
                    config=config,
                    save_dir=save_dir,
                    target_column="accent_label" if target == "accent" else "word_label",
                )
            st.success("Training complete.")
            metrics = {
                key: report[key]
                for key in ("accuracy", "precision", "recall", "f1", "wer")
            }
            st.json(metrics)
            st.caption("Per-accent accuracy")
            st.json(report["per_accent"])
        except Exception as exc:
            st.error(str(exc))
