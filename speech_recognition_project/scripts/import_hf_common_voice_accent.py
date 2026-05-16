"""Import accent-labelled Swahili rows from Hugging Face Common Voice shards.

This targets the public `fixie-ai/common_voice_17_0` mirror because Mozilla Data
Collective downloads require authenticated browser/API access.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import requests


HF_REPO = "fixie-ai/common_voice_17_0"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
HF_API_BASE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
METADATA_COLUMNS = [
    "file_path",
    "word_label",
    "accent_label",
    "speaker_id",
    "duration_sec",
    "split",
]

VARIANT_MAP = {
    "kimvita (ke) - central dialect": "coastal",
    "kiswahili cha bara ya kenya": "upcountry",
}


def _normalize_variant(value: object) -> str:
    return str(value or "").strip().lower()


def _normalize_word(value: object) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^\w\s'-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.split()[0] if text else ""


def _list_parquet_paths(split: str) -> list[str]:
    response = requests.get(f"{HF_API_BASE}/sw/{split}", timeout=30)
    response.raise_for_status()
    paths: list[str] = []
    for directory in response.json():
        if directory.get("type") != "directory":
            continue
        dir_path = directory["path"]
        shard_response = requests.get(f"{HF_API_BASE}/{dir_path}", timeout=30)
        shard_response.raise_for_status()
        for item in shard_response.json():
            path = item.get("path", "")
            if path.endswith(".parquet"):
                paths.append(path)
    return paths


def _download(path: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / Path(path).name
    if output.exists() and output.stat().st_size > 0:
        return output

    url = f"{HF_BASE}/{path}"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return output


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def _write_audio(audio_value: object, output_dir: Path, fallback_name: str) -> Path | None:
    if not isinstance(audio_value, dict):
        return None
    data = audio_value.get("bytes")
    source_path = str(audio_value.get("path") or fallback_name)
    if not data:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(Path(source_path).name or fallback_name)
    if not filename.endswith(".mp3"):
        filename = f"{filename}.mp3"
    output = output_dir / filename
    if not output.exists():
        output.write_bytes(data)
    return output


def _row_to_metadata(row, audio_path: Path, data_dir: Path, split: str, accent: str) -> dict[str, object]:
    try:
        file_path = audio_path.relative_to(data_dir).as_posix()
    except ValueError:
        file_path = str(audio_path)
    return {
        "file_path": file_path,
        "word_label": _normalize_word(row.get("sentence", "")),
        "accent_label": accent,
        "speaker_id": str(row.get("client_id") or "common_voice"),
        "duration_sec": 0.0,
        "split": "test" if split in {"test", "validation"} else "train",
    }


def import_rows(
    splits: list[str],
    data_dir: Path,
    output_csv: Path,
    cache_dir: Path,
    max_per_accent: int | None,
) -> pd.DataFrame:
    counts = {accent: 0 for accent in set(VARIANT_MAP.values())}
    rows: list[dict[str, object]] = []

    for split in splits:
        print(f"Listing {split} shards...")
        for shard_path in _list_parquet_paths(split):
            if max_per_accent is not None and all(count >= max_per_accent for count in counts.values()):
                break
            print(f"Reading {shard_path}")
            parquet_path = _download(shard_path, cache_dir)
            frame = pd.read_parquet(
                parquet_path,
                columns=["client_id", "path", "audio", "sentence", "variant"],
            )
            for _, row in frame.iterrows():
                accent = VARIANT_MAP.get(_normalize_variant(row.get("variant", "")))
                if not accent:
                    continue
                if max_per_accent is not None and counts[accent] >= max_per_accent:
                    continue
                word = _normalize_word(row.get("sentence", ""))
                if not word:
                    continue
                audio_path = _write_audio(
                    row.get("audio"),
                    data_dir / "common_voice_17_sw" / accent,
                    str(row.get("path") or f"{accent}_{counts[accent]}.mp3"),
                )
                if audio_path is None:
                    continue
                row_split = "test" if counts[accent] % 5 == 0 else "train"
                rows.append(_row_to_metadata(row, audio_path, data_dir, row_split, accent))
                counts[accent] += 1
            print("Counts:", counts)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    frame.to_csv(output_csv, index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Common Voice accent rows from Hugging Face.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/common_voice_accent_metadata.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/.hf_common_voice_cache"))
    parser.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    parser.add_argument("--max-per-accent", type=int, default=150)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = import_rows(
        splits=args.splits,
        data_dir=args.data_dir,
        output_csv=args.output,
        cache_dir=args.cache_dir,
        max_per_accent=args.max_per_accent,
    )
    print(f"Wrote {len(frame)} rows to {args.output}")
    if not frame.empty:
        print(frame["accent_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
