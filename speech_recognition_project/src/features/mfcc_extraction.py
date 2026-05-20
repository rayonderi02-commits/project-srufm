"""MFCC feature extraction with optional temporal statistics."""

from __future__ import annotations

import numpy as np


class FeatureExtractor:
    """Extract MFCC, delta, and delta-delta features from preprocessed audio."""

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 40,
        sr: int = 16000,
        aggregation: str = "mean",
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr
        if aggregation not in {"mean", "temporal_stats"}:
            raise ValueError("aggregation must be 'mean' or 'temporal_stats'.")
        self.aggregation = aggregation

    @property
    def feature_dim(self) -> int:
        base_dim = self.n_mfcc * 3
        if self.aggregation == "temporal_stats":
            return base_dim * 7
        return base_dim

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        import librosa

        return librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def extract_delta(self, mfcc: np.ndarray) -> np.ndarray:
        import librosa

        return librosa.feature.delta(mfcc, order=1)

    def extract_delta_delta(self, mfcc: np.ndarray) -> np.ndarray:
        import librosa

        return librosa.feature.delta(mfcc, order=2)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            raise ValueError("Cannot extract features from empty audio.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf values.")

        mfcc = self.extract_mfcc(audio)
        delta = self.extract_delta(mfcc)
        delta2 = self.extract_delta_delta(mfcc)
        combined = np.vstack([mfcc, delta, delta2])

        if self.aggregation == "temporal_stats":
            feature_vector = self._temporal_stats(combined)
        else:
            feature_vector = np.mean(combined, axis=1)

        assert feature_vector.shape == (self.feature_dim,), (
            f"Expected shape ({self.feature_dim},), got {feature_vector.shape}"
        )
        assert np.isfinite(feature_vector).all(), (
            "Feature vector contains NaN or Inf after extraction."
        )
        return feature_vector.astype(np.float64)

    def _temporal_stats(self, features: np.ndarray) -> np.ndarray:
        """Preserve coarse word timing with global stats and segment means."""
        chunks = np.array_split(features, 3, axis=1)
        parts = [
            np.mean(features, axis=1),
            np.std(features, axis=1),
            np.min(features, axis=1),
            np.max(features, axis=1),
        ]
        parts.extend(np.mean(chunk, axis=1) for chunk in chunks)
        return np.concatenate(parts, axis=0)

    def extract_batch(self, audio_list: list[np.ndarray]) -> np.ndarray:
        features = [self.extract(audio) for audio in audio_list]
        return np.array(features)
