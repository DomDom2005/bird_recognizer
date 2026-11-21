"""Load a saved run and evaluate it (SPEC FR-10, §11.3).

A run is self-contained: ``config.yaml`` holds the geometry, ``metrics.json`` the model
family/options, ``best.pt`` the weights. Evaluation rebuilds exactly that model and cuts
windows with exactly that geometry — it never borrows the current ``configs/`` values.
Window metrics come from the window predictions, recording metrics from the aggregated
probabilities (``mean`` is the headline; ``vote`` is written next to it).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from ..config import Config
from ..data.audio import load_audio
from ..data.dataset import WindowDataset, build_window_index
from ..data.spectrogram import mel_spectrogram
from ..data.windows import iter_window_signals, plan_windows, valid_frames
from ..errors import DataError, ModelError
from ..models_registry import build_model
from ..train.loop import forward_logits, predict_probs
from .aggregate import coverage_curve, vote_by_group
from .metrics import (
    accuracy,
    aggregate_by_group,
    balanced_accuracy,
    confusion,
    macro_f1,
    per_class_metrics,
    top_k_accuracy,
)

LOGGER = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
OWN_DIRNAME = "own"


@dataclass(frozen=True)
class RecordingScores:
    """Window probabilities plus the mean-aggregated recording probabilities."""

    order: list[str]
    scores: np.ndarray
    truth: list[int]
    window_scores: np.ndarray
    window_groups: list[str]
    window_truth: list[int]


@dataclass(frozen=True)
class RunBundle:
    """Everything needed to evaluate one run on disk."""

    run_id: str
    directory: Path
    config: Config
    model: nn.Module
    device: torch.device
    data_dir: Path
    model_name: str
    options: dict[str, Any]


def resolve_run_dir(runs_dir: Path, run: str | Path) -> Path:
    """Accept a run id (looked up in ``runs/``) or an explicit directory."""
    candidate = Path(run)
    if candidate.is_dir():
        return candidate
    fallback = Path(runs_dir) / str(run)
    if fallback.is_dir():
        return fallback
    raise DataError(f"brak katalogu runu '{run}' w {runs_dir}")


def _apply_snapshot(
    base: Config, snapshot: dict[str, Any], data_dir: Path
) -> Config:
    """Rebuild the run geometry from its config snapshot (no re-validation needed)."""
    audio_snapshot = snapshot.get("audio") or {}
    audio_fields = {
        key: audio_snapshot[key]
        for key in ("sample_rate", "highpass_hz", "highpass_order", "window_s", "hop_s")
        if key in audio_snapshot
    }
    spectrogram_snapshot = snapshot.get("spectrogram") or {}
    spectrogram_fields = {
        key: spectrogram_snapshot[key]
        for key in (
            "n_fft",
            "win_length",
            "hop_length",
            "n_mels",
            "fmin",
            "fmax",
            "power",
            "top_db",
            "normalise",
        )
        if key in spectrogram_snapshot
    }
    return replace(
        base,
        audio=replace(base.audio, **audio_fields),
        spectrogram=replace(base.spectrogram, **spectrogram_fields),
        paths=replace(base.paths, data_dir=Path(data_dir)),
    )


def load_run(
    config: Config,
    run: str | Path,
    *,
    device: torch.device,
    runs_dir: Path | None = None,
) -> RunBundle:
    """Load config snapshot, metrics and weights of one training run."""
    runs_dir = Path(runs_dir) if runs_dir is not None else Path(config.paths.runs_dir)
    directory = resolve_run_dir(runs_dir, run)
    metrics_path = directory / "metrics.json"
    config_path = directory / "config.yaml"
    if not metrics_path.is_file() or not config_path.is_file():
        raise DataError(
            f"{directory}: brak metrics.json lub config.yaml — to nie jest kompletny run"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    snapshot = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_dir = Path(
        metrics.get("data_dir")
        or (snapshot.get("paths") or {}).get("data_dir")
        or config.paths.data_dir
    )
    run_config = _apply_snapshot(config, snapshot, data_dir)
    model_name = str(metrics.get("model") or "")
    if model_name == "mfcc":
        raise ModelError(
            "eval obsługuje runy neuronowe (cnn/crnn/transfer); "
            "MFCC ma model.joblib i nie ma best.pt"
        )
    checkpoint = directory / "best.pt"
    if not checkpoint.is_file():
        raise ModelError(f"{directory}: brak best.pt — run nie ma zapisanego modelu")
    options = dict(metrics.get("options") or {})
    model = build_model(model_name, len(run_config.species), **options).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return RunBundle(
        run_id=directory.name,
        directory=directory,
        config=run_config,
        model=model,
        device=device,
        data_dir=data_dir,
        model_name=model_name,
        options=options,
    )


def species_names(config: Config) -> dict[int, str]:
    return {index: species.pl for index, species in enumerate(config.species)}


def _labels(config: Config) -> list[int]:
    return list(range(len(config.species)))


def split_recording_scores(bundle: RunBundle, split: str) -> RecordingScores:
    """Predict every window of a split and aggregate to recording level."""
    refs = build_window_index(bundle.config, bundle.data_dir, split=split)
    if not refs:
        raise DataError(
            f"brak okien w podziale '{split}' w {bundle.data_dir} — sprawdź cache i splits.json"
        )
    dataset = WindowDataset(refs, with_lengths=True)
    probs = predict_probs(bundle.model, dataset, bundle.device)
    groups = dataset.groups()
    order, matrix = aggregate_by_group(probs, groups)
    truth_map: dict[str, int] = {}
    for label, group in zip(dataset.labels, groups, strict=True):
        truth_map.setdefault(group, int(label))
    truth = [truth_map[group] for group in order]
    return RecordingScores(
        order=order,
        scores=matrix,
        truth=truth,
        window_scores=probs,
        window_groups=groups,
        window_truth=[int(label) for label in dataset.labels],
    )


def _model_probs(
    model: nn.Module,
    windows: np.ndarray,
    lengths: list[int],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).unsqueeze(1).to(device)
            batch_lengths = torch.tensor(
                lengths[start : start + batch_size], dtype=torch.long, device=device
            )
            logits = forward_logits(model, batch, batch_lengths)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks)


def own_recording_scores(bundle: RunBundle, own_dir: Path | None = None) -> RecordingScores | None:
    """Cut and score ``data/own/<gatunek>/*`` files; ``None`` when no own data exists."""
    base = own_dir or Path(bundle.config.paths.data_dir) / OWN_DIRNAME
    if not base.is_dir():
        return None
    labels = bundle.config.species_labels()
    order: list[str] = []
    scores: list[np.ndarray] = []
    truth: list[int] = []
    window_scores: list[np.ndarray] = []
    window_groups: list[str] = []
    window_truth: list[int] = []
    for species_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        species_pl = species_dir.name
        if species_pl not in labels:
            LOGGER.warning("pomijam '%s' — nie ma go w configs/species.yaml", species_pl)
            continue
        label = labels[species_pl]
        for path in sorted(
            file for file in species_dir.iterdir() if file.suffix.lower() in AUDIO_SUFFIXES
        ):
            try:
                signal = load_audio(path, bundle.config.audio)
                plan = plan_windows(signal, bundle.config.audio, bundle.config.gating)
            except DataError as exc:
                LOGGER.warning("%s: %s", path, exc)
                continue
            frames: list[np.ndarray] = []
            lengths: list[int] = []
            for index, window in iter_window_signals(signal, plan, only_kept=True):
                frames.append(
                    mel_spectrogram(
                        window,
                        bundle.config.audio.sample_rate,
                        bundle.config.spectrogram,
                    )
                )
                lengths.append(
                    valid_frames(plan.lengths[index], bundle.config.spectrogram.hop_length)
                )
            if not frames:
                LOGGER.warning("%s: brak okien po bramkowaniu ciszy", path)
                continue
            array = np.stack(frames).astype(np.float32)
            probs = _model_probs(bundle.model, array, lengths, bundle.device)
            recording_id = f"{species_pl}/{path.name}"
            order.append(recording_id)
            scores.append(probs.mean(axis=0))
            truth.append(label)
            window_scores.append(probs)
            window_groups.extend([recording_id] * len(frames))
            window_truth.extend([label] * len(frames))
    if not order:
        return None
    return RecordingScores(
        order=order,
        scores=np.stack(scores),
        truth=truth,
        window_scores=np.concatenate(window_scores),
        window_groups=window_groups,
        window_truth=window_truth,
    )


def _recording_metrics(
    truth: list[int],
    scores: np.ndarray,
    *,
    labels: list[int],
) -> dict[str, Any]:
    predictions = scores.argmax(axis=1)
    return {
        "n": int(len(truth)),
        "macro_f1": round(float(macro_f1(truth, predictions, labels=labels)), 6),
        "accuracy": round(float(accuracy(truth, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy(truth, predictions, labels=labels)), 6),
        "top1": round(float(accuracy(truth, predictions)), 6),
        "top3": _rounded(top_k_accuracy(truth, scores, k=3, labels=labels)),
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _window_metrics(
    truth: list[int],
    probs: np.ndarray,
    *,
    labels: list[int],
) -> dict[str, Any]:
    predictions = probs.argmax(axis=1)
    return {
        "n": int(len(truth)),
        "macro_f1": round(float(macro_f1(truth, predictions, labels=labels)), 6),
        "accuracy": round(float(accuracy(truth, predictions)), 6),
        "top3": _rounded(top_k_accuracy(truth, probs, k=3, labels=labels)),
    }


def evaluate_split(bundle: RunBundle, split: str) -> dict[str, Any]:
    """Evaluate ``val``/``test``/``own``; writes ``runs/<run>/eval_<split>/``."""
    if split == "own":
        scores = own_recording_scores(bundle)
        if scores is None:
            raise DataError(
                f"brak nagrań własnych w {Path(bundle.config.paths.data_dir) / OWN_DIRNAME}"
                "/<gatunek>/ (*.wav, *.mp3, *.flac, *.ogg). To zadanie człowieka (SPEC §16)."
            )
    else:
        scores = split_recording_scores(bundle, split)
    labels = _labels(bundle.config)
    names = species_names(bundle.config)
    mean_metrics = _recording_metrics(scores.truth, scores.scores, labels=labels)
    vote_order, vote_predictions = vote_by_group(scores.window_scores, scores.window_groups)
    vote_truth = [scores.truth[scores.order.index(recording)] for recording in vote_order]
    vote_metrics = {
        "n": int(len(vote_truth)),
        "macro_f1": round(float(macro_f1(vote_truth, vote_predictions, labels=labels)), 6),
        "accuracy": round(float(accuracy(vote_truth, vote_predictions)), 6),
        "top1": round(float(accuracy(vote_truth, vote_predictions)), 6),
    }
    predictions = scores.scores.argmax(axis=1)
    per_class = per_class_metrics(scores.truth, predictions, labels=labels, names=names)
    matrix = confusion(scores.truth, predictions, labels=labels)
    payload: dict[str, Any] = {
        "run_id": bundle.run_id,
        "split": split,
        "model": bundle.model_name,
        "options": dict(bundle.options),
        "seed": bundle.config.seed,
        "window_s": bundle.config.audio.window_s,
        "aug": bundle.config.augment.preset,
        "data_dir": str(bundle.data_dir),
        "n_recordings": int(len(scores.order)),
        "n_windows": int(len(scores.window_groups)),
        "recording": {**mean_metrics, "aggregation": "mean"},
        "vote": vote_metrics,
        "window": _window_metrics(scores.window_truth, scores.window_scores, labels=labels),
        "coverage": coverage_curve(scores.truth, scores.scores),
    }
    out_dir = bundle.directory / f"eval_{split}"
    _write_outputs(
        out_dir,
        payload=payload,
        per_class=per_class,
        matrix=matrix,
        names=names,
        scores=scores,
    )
    LOGGER.info(
        "eval %s/%s: nagrania=%s, mean macro-F1=%.3f, vote macro-F1=%.3f",
        bundle.run_id,
        split,
        payload["n_recordings"],
        mean_metrics["macro_f1"],
        vote_metrics["macro_f1"],
    )
    return payload


def _write_outputs(
    out_dir: Path,
    *,
    payload: dict[str, Any],
    per_class: list[dict[str, Any]],
    matrix: np.ndarray,
    names: dict[int, str],
    scores: RecordingScores,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (out_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "species", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(per_class)
    species = [names[label] for label in sorted(names)]
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\prediction", *species])
        for row_index, row in enumerate(matrix):
            writer.writerow([species[row_index], *[int(value) for value in row]])
    predictions = scores.scores.argmax(axis=1)
    with (out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recording_id", "truth", "predicted", "confidence", "p_true"])
        for index, recording in enumerate(scores.order):
            writer.writerow(
                [
                    recording,
                    names.get(scores.truth[index], scores.truth[index]),
                    names.get(int(predictions[index]), int(predictions[index])),
                    round(float(scores.scores[index].max()), 6),
                    round(float(scores.scores[index][scores.truth[index]]), 6),
                ]
            )
    _save_confusion_png(
        out_dir / "confusion_matrix.png",
        matrix,
        species,
        title=f"{payload['run_id']} / {payload['split']}",
    )


def _save_confusion_png(
    path: Path, matrix: np.ndarray, species: list[str], *, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("predykcja")
    axis.set_ylabel("prawda")
    axis.set_xticks(range(len(species)))
    axis.set_xticklabels(species, rotation=90, fontsize=7)
    axis.set_yticks(range(len(species)))
    axis.set_yticklabels(species, fontsize=7)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            if value:
                axis.text(column, row, str(value), ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

