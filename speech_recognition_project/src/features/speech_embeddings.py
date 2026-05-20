"""Pretrained speech embedding extraction for low-resource word classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechEmbeddingConfig:
    """Configuration for a Hugging Face speech encoder."""

    model_name: str = "facebook/wav2vec2-xls-r-300m"
    sr: int = 16000


class PretrainedSpeechEmbeddingExtractor:
    """Extract mean-pooled hidden-state embeddings from a pretrained speech model."""

    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", sr: int = 16000):
        self.model_name = model_name
        self.sr = sr
        self._processor = None
        self._model = None
        self._torch = None

    @property
    def feature_dim(self) -> int:
        self._load()
        return int(self._model.config.hidden_size)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Pretrained speech embeddings require torch and transformers. "
                "Install them with: pip install torch transformers"
            ) from exc

        self._torch = torch
        self._processor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Return one fixed-size embedding for a mono 16 kHz waveform."""
        if len(audio) == 0:
            raise ValueError("Cannot extract embeddings from empty audio.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf values.")

        self._load()
        torch = self._torch
        waveform = np.asarray(audio, dtype=np.float32)
        inputs = self._processor(
            waveform,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state

            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                frame_mask = self._get_feature_vector_attention_mask(
                    hidden.shape[1],
                    attention_mask,
                )
                frame_mask = frame_mask.to(hidden.device).unsqueeze(-1)
                pooled = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)

        return pooled.squeeze(0).cpu().numpy().astype(np.float64)

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        if hasattr(self._model, "_get_feature_vector_attention_mask"):
            return self._model._get_feature_vector_attention_mask(
                feature_vector_length,
                attention_mask,
            )
        return attention_mask.new_ones((attention_mask.shape[0], feature_vector_length))
