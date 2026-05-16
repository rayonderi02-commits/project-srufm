"""Build accent metadata from Common Voice variants plus local recordings.

Mappings:
- Common Voice `sw-kimvita` -> coastal
- Common Voice `sw-barake` -> upcountry
- local USB recordings keep their labelled accent values

KenSpeech can be added only as a weak `nairobi` source when explicitly enabled,
because it is not cleanly labelled as Nairobi accent data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = [
    "file_path",
    "word_label",
    "accent_label",
    "speaker_id",
    "duration_sec",
    "split",
]

COMMON_VOICE_VARIANT_MAP = {
    "sw-kimvita": "coastal",
    "sw-barake": "upcountry",
}


def _normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text)


def _safe_id(value: object, fallback: str) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text or fallback


def _duration_lookup(common_voice_root: Path) -> dict[str, float]:
    duration_path = common_voice_root / "clip_durations.tsv"
    if not duration_path.exists():
        return {}
    frame = pd.read_csv(duration_path, sep="\t")
    if frame.empty:
        return {}

    clip_col = next((c for c in ("clip", "path", "file_path") if c in frame.columns), None)
    duration_col = next(
        (c for c in ("duration[ms]", "duration_ms", "duration", "duration_sec") if c in frame.columns),
        None,
    )
    if clip_col is None or duration_col is None:
        return {}

    durations = {}
    for _, row in frame.iterrows():
        clip = str(row[clip_col])
        raw_duration = float(row[duration_col])
        duration = raw_duration / 1000.0 if raw_duration > 100 else raw_duration
        durations[Path(clip).name] = duration
    return durations


def _load_common_voice(
    common_voice_root: Path,
    data_dir: Path,
    max_rows_per_accent: int | None,
    words: set[str] | None,
) -> list[dict[str, object]]:
    if not common_voice_root.exists():
        raise FileNotFoundError(f"Common Voice root not found: {common_voice_root}")

    durations = _duration_lookup(common_voice_root)
    rows: list[dict[str, object]] = []
    tsv_paths = [
        path
        for path in (
            common_voice_root / "train.tsv",
            common_voice_root / "dev.tsv",
            common_voice_root / "test.tsv",
            common_voice_root / "validated.tsv",
        )
        if path.exists()
    ]
    if not tsv_paths:
        raise FileNotFoundError(f"No Common Voice TSV files found in {common_voice_root}")

    per_accent_counts = {accent: 0 for accent in COMMON_VOICE_VARIANT_MAP.values()}
    for tsv_path in tsv_paths:
        split_from_file = _split_name(tsv_path.name)
        frame = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
        if "variant" not in frame.columns or "path" not in frame.columns:
            continue
        for index, row in frame.iterrows():
            variant = str(row.get("variant", "")).strip().lower()
            accent = COMMON_VOICE_VARIANT_MAP.get(variant)
            if not accent:
                continue
            if max_rows_per_accent is not None and per_accent_counts[accent] >= max_rows_per_accent:
                continue

            sentence = _normalize_text(row.get("sentence", ""))
            if not sentence:
                continue
            word = sentence.split()[0]
            if words and word not in words:
                continue

            clip_name = str(row.get("path", "")).strip()
            audio_path = common_voice_root / "clips" / clip_name
            if not audio_path.exists():
                continue

            try:
                file_path = audio_path.relative_to(data_dir).as_posix()
            except ValueError:
                file_path = str(audio_path)

            split = split_from_file
            if split == "validated":
                split = "test" if index % 5 == 0 else "train"

            rows.append(
                {
                    "file_path": file_path,
                    "word_label": word,
                    "accent_label": accent,
                    "speaker_id": _safe_id(row.get("client_id", ""), f"cv_{index}"),
                    "duration_sec": float(durations.get(Path(clip_name).name, 0.0)),
                    "split": split,
                }
            )
            per_accent_counts[accent] += 1

    return rows


def _split_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if stem == "dev":
        return "test"
    if stem in {"train", "test"}:
        return stem
    return "validated"


def _load_local_metadata(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    missing = set(METADATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Local metadata missing columns: {sorted(missing)}")
    frame = frame[frame["accent_label"].isin(["coastal", "nairobi", "upcountry"])]
    return frame[METADATA_COLUMNS].to_dict(orient="records")


def _load_words(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    words = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return words or None


def _dedupe(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    output = []
    for row in rows:
        key = hashlib.sha1(str(row["file_path"]).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hybrid accent metadata.")
    parser.add_argument("--common-voice-root", type=Path)
    parser.add_argument(
        "--local-metadata",
        type=Path,
        default=Path("data/accent_metadata.csv"),
        help="Metadata from USB microphone recordings.",
    )
    parser.add_argument(
        "--extra-metadata",
        type=Path,
        action="append",
        default=[],
        help="Additional trainer-compatible metadata CSV to merge.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory used by the trainer to resolve audio paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hybrid_accent_metadata.csv"),
    )
    parser.add_argument("--words-file", type=Path)
    parser.add_argument("--max-common-voice-per-accent", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    words = _load_words(args.words_file)
    rows: list[dict[str, object]] = []

    if args.common_voice_root:
        rows.extend(
            _load_common_voice(
                common_voice_root=args.common_voice_root,
                data_dir=args.data_dir,
                max_rows_per_accent=args.max_common_voice_per_accent,
                words=words,
            )
        )
    rows.extend(_load_local_metadata(args.local_metadata))
    for metadata_path in args.extra_metadata:
        rows.extend(_load_local_metadata(metadata_path))
    rows = _dedupe(rows)

    if not rows:
        raise ValueError("No rows found. Add Common Voice data or local recordings first.")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    frame.to_csv(output, index=False)

    print(f"Wrote {len(frame)} rows to {output}")
    print(frame["accent_label"].value_counts().to_string())
    print("Splits:")
    print(frame["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
