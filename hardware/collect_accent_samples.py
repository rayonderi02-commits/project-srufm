"""Collect labelled accent samples from the Raspberry Pi USB microphone."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASR_ROOT = PROJECT_ROOT / "speech_recognition_project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.accent_hardware_runner import SAMPLE_RATE, record_audio  # noqa: E402


VALID_ACCENTS = ("coastal", "nairobi", "upcountry")
METADATA_COLUMNS = (
    "file_path",
    "word_label",
    "accent_label",
    "speaker_id",
    "duration_sec",
    "split",
)


def _metadata_path(path: Path | None) -> Path:
    return path or ASR_ROOT / "data" / "accent_metadata.csv"


def _relative_to_data_dir(path: Path, data_dir: Path) -> str:
    return path.resolve().relative_to(data_dir.resolve()).as_posix()


def _ensure_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()


def _append_row(path: Path, row: dict[str, str | float]) -> None:
    _ensure_metadata(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record labelled USB microphone samples for accent training."
    )
    parser.add_argument("--accent", required=True, choices=VALID_ACCENTS)
    parser.add_argument(
        "--word",
        required=True,
        help="Prompt word or phrase being spoken, for example 'ndiyo'.",
    )
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="Anonymous speaker id, for example coastal_001.",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional PyAudio USB microphone device index.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ASR_ROOT / "data" / "raw",
        help="Directory where recorded WAV files will be stored.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Metadata CSV to append. Defaults to "
            "speech_recognition_project/data/accent_metadata.csv."
        ),
    )
    parser.add_argument(
        "--test-probability",
        type=float,
        default=0.2,
        help="Probability that each sample is marked as test split.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata_path = _metadata_path(args.metadata)
    data_dir = args.data_dir
    output_dir = data_dir / args.accent / args.speaker_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Recording {args.samples} sample(s) for accent={args.accent}, "
        f"word={args.word}, speaker={args.speaker_id}"
    )
    print("Speak clearly after each countdown.")

    for sample_number in range(1, args.samples + 1):
        for count in (3, 2, 1):
            print(count)
            time.sleep(0.7)
        print("Speak now...")

        audio = record_audio(duration=args.duration, device_index=args.device_index)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_word = "".join(ch if ch.isalnum() else "_" for ch in args.word.lower())
        filename = (
            f"{args.speaker_id}_{args.accent}_{safe_word}_"
            f"{timestamp}_{sample_number:03d}.wav"
        )
        audio_path = output_dir / filename
        sf.write(audio_path, audio, SAMPLE_RATE)

        split = "test" if random.random() < args.test_probability else "train"
        row = {
            "file_path": _relative_to_data_dir(audio_path, data_dir),
            "word_label": args.word,
            "accent_label": args.accent,
            "speaker_id": args.speaker_id,
            "duration_sec": f"{len(audio) / SAMPLE_RATE:.3f}",
            "split": split,
        }
        _append_row(metadata_path, row)
        print(f"Saved {audio_path} [{split}]")

    print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
