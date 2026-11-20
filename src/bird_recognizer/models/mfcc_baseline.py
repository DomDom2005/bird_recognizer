"""MFCC + statistics baseline: SVM or Random Forest (SPEC FR-5).

Features per window: 40 MFCC coefficients, then mean and standard deviation over
time (80 numbers). Standardisation is fitted **on the training split only**.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torchaudio.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import Config
from ..data.audio import load_audio
from ..data.dataset import META_FILENAME, WindowRef, kept_windows, raw_audio_path
from ..errors import ModelError

LOGGER = logging.getLogger(__name__)

N_MFCC = 40
_CLASSIFIER_KINDS = ("svm", "rf")


@lru_cache(maxsize=4)
def _mfcc_transform(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> transforms.MFCC:
    return transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": N_MFCC,
            "f_min": fmin,
            "f_max": fmax,
            "center": True,
            "power": 2.0,
        },
    )


def mfcc_vector(window: np.ndarray, config: Config) -> np.ndarray:
    """80-dimensional feature vector: MFCC mean + standard deviation over time."""
    transform = _mfcc_transform(
        config.audio.sample_rate,
        config.spectrogram.n_fft,
        config.spectrogram.hop_length,
        config.spectrogram.fmin,
        config.spectrogram.fmax,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
    coefficients = transform(tensor).numpy()
    if coefficients.ndim != 2:
        raise ModelError(f"MFCC ma nieoczekiwany kształt: {coefficients.shape}")
    return np.concatenate(
        [coefficients.mean(axis=1), coefficients.std(axis=1)]
    ).astype(np.float32)


def build_classifier(kind: str, seed: int = 42) -> Pipeline:
    """Scaler + classifier pipeline; the scaler sees only the training features."""
    if kind == "svm":
        classifier = SVC(
            kernel="rbf",
            class_weight="balanced",
            cache_size=512,
            decision_function_shape="ovr",
        )
    elif kind == "rf":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ModelError(f"nieznany klasyfikator '{kind}'; dostępne: {_CLASSIFIER_KINDS}")
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


@dataclass
class MFCCBaseline:
    """Fitted baseline with per-class scoring and prediction helpers."""

    model: Pipeline
    kind: str

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        kind: str = "svm",
        seed: int = 42,
    ) -> MFCCBaseline:
        if features.shape[0] == 0:
            raise ModelError("brak cech treningowych dla baseline'u MFCC")
        if len(np.unique(labels)) < 2:
            raise ModelError("baseline potrzebuje co najmniej 2 klas w trainie")
        pipeline = build_classifier(kind, seed)
        pipeline.fit(features, labels)
        return cls(model=pipeline, kind=kind)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.model.classes_)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Per-class scores aligned with ``classes_`` (probabilities when available)."""
        classifier = self.model[-1]
        if hasattr(classifier, "predict_proba"):
            return np.asarray(self.model.predict_proba(features), dtype=np.float64)
        decision = np.asarray(classifier.decision_function(features), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        return decision

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.scores(features), axis=1)]


def features_from_refs(
    config: Config,
    data_dir,
    refs: list[WindowRef],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """MFCC features for cached windows, re-cut from the raw audio.

    The geometry comes from the cache metadata, so the baseline sees exactly the
    same windows and gating decisions as the CNN.
    """
    if not refs:
        return np.zeros((0, 2 * N_MFCC), dtype=np.float32), np.zeros(0, dtype=np.int64), []

    grouped: dict[str, list[WindowRef]] = defaultdict(list)
    for ref in refs:
        grouped[ref.recording_id].append(ref)

    target = int(round(config.audio.window_s * config.audio.sample_rate))
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for recording_id in sorted(grouped):
        recording_refs = grouped[recording_id]
        audio_file = raw_audio_path(data_dir, recording_id)
        if audio_file is None:
            LOGGER.warning("%s: brak pliku audio — pomijam w baseline MFCC", recording_id)
            continue
        meta_file = recording_refs[0].cache_dir / META_FILENAME
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("%s: nie mogę odczytać meta (%s)", recording_id, exc)
            continue
        geometry = kept_windows(meta)
        if not geometry:
            continue
        signal = load_audio(audio_file, config.audio)
        for ref in recording_refs:
            if ref.window >= len(geometry):
                continue
            start, length = geometry[ref.window]
            window = signal[start : start + length]
            if window.size < target:
                window = np.pad(window, (0, target - window.size))
            features.append(mfcc_vector(window, config))
            labels.append(int(ref.label))
            groups.append(recording_id)

    if not features:
        raise ModelError("brak cech MFCC — sprawdź cache i data/raw")
    return np.stack(features), np.asarray(labels, dtype=np.int64), groups
