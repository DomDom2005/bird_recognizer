"""Command line interface (SPEC §8)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import torch

from . import ablation
from .config import DEFAULT_CONFIG_PATH, Config, TrainingConfig, get_api_key, load_config
from .data.dataset import (
    WindowDataset,
    build_cache,
    build_window_index,
    list_cache,
    verify_cache,
)
from .data.splits import load_splits, make_and_save, splits_path
from .errors import AppError, DataError, ModelError
from .eval import error_analysis as error_analysis_module
from .eval import evaluate as evaluate_module
from .eval.aggregate import coverage_curve, vote_by_group
from .eval.metrics import accuracy, aggregate_by_group, macro_f1, top_k_accuracy
from .eval.report import select_rows, write_report
from .logging_setup import configure_console, setup_logging
from .models.mfcc_baseline import MFCCBaseline, features_from_refs
from .models_registry import build_model
from .train.loop import set_seed, train_model
from .train.runner import Run, base_index_row, load_index
from .xc.catalog import build_catalog, catalog_path, load_catalog
from .xc.client import XCClient
from .xc.download import download_audio

LOGGER = logging.getLogger(__name__)

#: SVM is O(n^2); cap the training set so a full run stays practical on CPU.
MAX_SVM_TRAIN_WINDOWS = 20_000
SMOKE_LIMIT_PER_SPECIES = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree: global options + ``data`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="bird_recognizer",
        description="Klasyfikacja gatunku ptaka z nagrania: dane, trening, ewaluacja.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--smoke", action="store_true", help="szybki przebieg na podzbiorze")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="katalog nagrań, pobieranie audio, podział")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    catalog = data_sub.add_parser("catalog", help="pobierz metadane z xeno-canto")
    catalog.add_argument("--species", nargs="*", default=None, help="nazwy PL lub łacińskie")
    catalog.add_argument("--refresh", action="store_true", help="odbuduj katalog od zera")

    download = data_sub.add_parser("download", help="pobierz pliki audio")
    download.add_argument("--species", nargs="*", default=None)
    download.add_argument("--limit", type=int, default=None)
    download.add_argument("--force", action="store_true")
    download.add_argument("--retry-failed", action="store_true")

    split = data_sub.add_parser("split", help="podział train/val/test")
    split.add_argument("--force", action="store_true", help="nadpisz istniejący podział")

    prep = sub.add_parser("prep", help="preprocessing: cache spektrogramów")
    prep_sub = prep.add_subparsers(dest="prep_command", required=True)
    cache = prep_sub.add_parser("cache", help="zbuduj cache okien log-mel")
    cache.add_argument("--species", nargs="*", default=None)
    cache.add_argument("--limit", type=int, default=None, help="pierwsze N nagrań z katalogu")
    cache.add_argument("--force", action="store_true", help="przelicz także aktualny cache")
    cache.add_argument("--verify", action="store_true", help="tylko sprawdź cache")

    train = sub.add_parser("train", help="trenowanie modeli")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    mfcc = train_sub.add_parser("mfcc", help="baseline MFCC + SVM/RF")
    mfcc.add_argument("--clf", choices=["svm", "rf"], default="svm")
    mfcc.add_argument(
        "--smoke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="szybki przebieg na podzbiorze (także po podkomendzie)",
    )
    cnn = train_sub.add_parser("cnn", help="mały CNN bez rekurencji")
    cnn.add_argument("--epochs", type=int, default=None, help="nadpisz liczbę epok")
    cnn.add_argument(
        "--aug",
        choices=["none", "basic", "full"],
        default=None,
        help="wariant augmentacji (domyślnie z configu)",
    )
    cnn.add_argument(
        "--smoke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="szybki przebieg na podzbiorze (także po podkomendzie)",
    )
    crnn = train_sub.add_parser("crnn", help="CRNN: CNN + BiGRU + pooling czasowy")
    crnn.add_argument("--pool", choices=["mean", "attention"], default="mean")
    crnn.add_argument("--rnn", choices=["gru", "lstm"], default="gru")
    crnn.add_argument("--epochs", type=int, default=None, help="nadpisz liczbę epok")
    crnn.add_argument(
        "--aug",
        choices=["none", "basic", "full"],
        default=None,
        help="wariant augmentacji (domyślnie z configu)",
    )
    crnn.add_argument(
        "--smoke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="szybki przebieg na podzbiorze (także po podkomendzie)",
    )
    transfer = train_sub.add_parser("transfer", help="transfer: ResNet18 / EfficientNet-B0")
    transfer.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    transfer.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="wagi ImageNet (domyślnie)",
    )
    transfer.add_argument(
        "--scratch",
        dest="pretrained",
        action="store_false",
        help="ten sam model bez wag ImageNet (ablacja)",
    )
    transfer.add_argument("--epochs", type=int, default=None, help="nadpisz liczbę epok")
    transfer.add_argument(
        "--aug",
        choices=["none", "basic", "full"],
        default=None,
        help="wariant augmentacji (domyślnie z configu)",
    )
    transfer.add_argument(
        "--smoke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="szybki przebieg na podzbiorze (także po podkomendzie)",
    )

    ablate = sub.add_parser("ablate", help="macierz ablacji jednoczynnikowych")
    ablate.add_argument("--group", choices=list(ablation.GROUPS), required=True)
    ablate.add_argument("--epochs", type=int, default=None, help="nadpisz budżet epok")
    ablate.add_argument(
        "--smoke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="szybki przebieg na podzbiorze (także po podkomendzie)",
    )

    report = sub.add_parser("report", help="tabela wyników z runs/index.csv")
    report.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run_id z runs/index.csv; pierwszy jest referencją dla Δ",
    )
    report.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments.md"),
        help="wyjściowy plik Markdown (domyślnie docs/experiments.md)",
    )

    evaluate = sub.add_parser("eval", help="metryki runu na poziomie okna i nagrania")
    evaluate.add_argument("--run", required=True, help="run_id albo ścieżka katalogu runu")
    evaluate.add_argument("--split", choices=["val", "test", "own"], default="test")

    aggregate = sub.add_parser("aggregate", help="agregacja mean vs vote + coverage")
    aggregate.add_argument("--run", required=True)
    aggregate.add_argument("--method", choices=["mean", "vote"], default="mean")
    aggregate.add_argument("--split", choices=["val", "test"], default="test")

    errors = sub.add_parser("error-analysis", help="pary mylące i najgorsze nagrania")
    errors.add_argument("--run", required=True)
    errors.add_argument("--split", choices=["val", "test"], default="test")
    errors.add_argument("--top", type=int, default=10, help="ile par pomyłek pokazać")

    demo = sub.add_parser("demo", help="demo Gradio: plik → top-3 gatunki")
    demo.add_argument("--run", required=True)
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument(
        "--smoke",
        action="store_true",
        default=argparse.SUPPRESS,
        help="nie uruchamiaj serwera: sprawdź ścieżkę na syntetycznym tonie",
    )

    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates: dict[str, object] = {}
    paths = config.paths
    paths_changed = False
    if args.data_dir is not None:
        paths = replace(paths, data_dir=args.data_dir)
        paths_changed = True
    if args.runs_dir is not None:
        paths = replace(paths, runs_dir=args.runs_dir)
        paths_changed = True
    if paths_changed:
        updates["paths"] = paths
    if args.seed is not None:
        updates["seed"] = args.seed
    return replace(config, **updates) if updates else config


def _cmd_catalog(config: Config, args: argparse.Namespace) -> int:
    client = XCClient(config)
    result = build_catalog(
        config,
        client,
        data_dir=config.paths.data_dir,
        species_filter=args.species,
        refresh=args.refresh,
    )
    print(f"Katalog: {len(result.entries)} wpisów -> {catalog_path(config.paths.data_dir)}")
    for species, info in result.queries.items():
        print(
            f"  {species}: {info['num_recordings']} nagrań w API, "
            f"zapisano {info['saved']} (stron: {info['pages']})"
        )
    for warning in result.warnings:
        print(f"UWAGA: {warning}")
    return 0


def _cmd_download(config: Config, args: argparse.Namespace) -> int:
    client = XCClient(config)
    report = download_audio(
        config,
        client,
        data_dir=config.paths.data_dir,
        species_filter=args.species,
        limit=args.limit,
        force=args.force,
        retry_failed=args.retry_failed,
    )
    print(
        "Pobieranie: "
        f"ok={len(report.succeeded)}, pominięte={len(report.skipped)}, "
        f"zredagowane={len(report.redacted)}, błędy={len(report.errors)}, "
        f"rozmiar={report.total_bytes / 1024 / 1024:.1f} MB"
    )
    if report.duplicates:
        print(f"UWAGA: {len(report.duplicates)} duplikatów plików (ten sam sha256)")
    if report.stopped:
        print(f"UWAGA: przerwano ({report.stopped}) — uruchom ponownie później")
        return 2
    return 0


def _cmd_cache(config: Config, args: argparse.Namespace) -> int:
    data_dir = config.paths.data_dir
    if args.verify:
        problems = verify_cache(config, data_dir)
        if problems:
            print(f"Cache NIEZGODNY ({len(problems)} problemów):")
            for problem in problems[:20]:
                print(f"  - {problem}")
            if len(problems) > 20:
                print(f"  ... i {len(problems) - 20} więcej")
            return 3
        print(f"Cache OK ({len(list_cache(data_dir))} nagrań)")
        return 0

    report = build_cache(
        config,
        data_dir,
        species=args.species,
        limit=args.limit,
        force=args.force,
    )
    print(
        f"Cache: przetworzone {report.processed}, pominięte {report.skipped}, "
        f"błędy {len(report.errors)}"
    )
    if report.skipped_reasons:
        reasons = ", ".join(
            f"{name}={count}" for name, count in sorted(report.skipped_reasons.items())
        )
        print(f"  pominięcia: {reasons}")
    print(
        f"  okna: zachowane {report.windows_kept}/{report.windows_total} "
        f"(odrzucone {report.rejected_ratio:.1%})"
    )
    print(f"  audio: {report.audio_seconds / 60:.1f} min")
    for species, stats in sorted(report.per_species.items()):
        ratio = 1.0 - stats["kept"] / stats["windows"] if stats["windows"] else 0.0
        print(f"  {species}: {stats['kept']}/{stats['windows']} okien (odrzucone {ratio:.1%})")
    for error in report.errors[:10]:
        print(f"  BŁĄD {error['id']}: {error['reason']}")
    if report.errors and report.processed == 0:
        return 3
    return 0


def _cmd_split(config: Config, args: argparse.Namespace) -> int:
    plan = make_and_save(config.paths.data_dir, seed=config.seed, force=args.force)
    for name in ("train", "val", "test"):
        counts = plan.counts.get(name, {})
        print(
            f"{name}: {counts.get('recordings', 0)} nagrań, "
            f"{counts.get('species', 0)} gatunków, {counts.get('groups', 0)} nagrywających"
        )
    for warning in plan.warnings:
        print(f"UWAGA: {warning}")
    return 0


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_plan(data_dir: Path):
    path = splits_path(data_dir)
    if not path.is_file():
        raise DataError(f"Brak {path}. Uruchom `bird_recognizer data split`.")
    return load_splits(path)


def _split_refs(
    config: Config, data_dir: Path, limit: int | None
) -> dict[str, list]:
    return {
        split: build_window_index(
            config, data_dir, split=split, limit_per_species=limit
        )
        for split in ("train", "val", "test")
    }


def _round_optional(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/d" if value is None else f"{value:.{digits}f}"


def _summarise_predictions(
    scores: np.ndarray,
    classes: np.ndarray,
    y_true: np.ndarray,
    groups: list[str],
) -> dict[str, float]:
    predictions = classes[np.argmax(scores, axis=1)]
    order, recording_scores = aggregate_by_group(scores, groups)
    truth: dict[str, int] = {}
    for label, group in zip(y_true.tolist(), groups, strict=True):
        truth.setdefault(group, int(label))
    recording_truth = [truth[group] for group in order]
    recording_predictions = classes[np.argmax(recording_scores, axis=1)]
    return {
        "window_n": int(len(y_true)),
        "window_macro_f1": round(float(macro_f1(y_true, predictions)), 6),
        "recording_n": int(len(order)),
        "recording_macro_f1": round(
            float(macro_f1(recording_truth, recording_predictions)), 6
        ),
        "recording_top1": round(
            float(accuracy(recording_truth, recording_predictions)), 6
        ),
        "recording_top3": _round_optional(
            top_k_accuracy(recording_truth, recording_scores, k=3, labels=classes)
        ),
    }


def _model_metrics(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    n_classes: int,
) -> dict[str, float]:
    from .train.loop import predict_probs

    probs = predict_probs(model, dataset, device)
    groups = dataset.groups()
    y_true = np.asarray(dataset.labels, dtype=np.int64)
    order, recording_scores = aggregate_by_group(probs, groups)
    truth: dict[str, int] = {}
    for label, group in zip(y_true.tolist(), groups, strict=True):
        truth.setdefault(group, int(label))
    recording_truth = [truth[group] for group in order]
    recording_predictions = np.argmax(recording_scores, axis=1)
    return {
        "window_n": int(len(y_true)),
        "window_macro_f1": round(float(macro_f1(y_true, probs.argmax(axis=1))), 6),
        "recording_n": int(len(order)),
        "recording_macro_f1": round(
            float(macro_f1(recording_truth, recording_predictions)), 6
        ),
        "recording_top1": round(
            float(accuracy(recording_truth, recording_predictions)), 6
        ),
        "recording_top3": _round_optional(
            top_k_accuracy(
                recording_truth,
                recording_scores,
                k=3,
                labels=list(range(n_classes)),
            )
        ),
    }


def _cmd_train_mfcc(config: Config, args: argparse.Namespace) -> int:
    data_dir = config.paths.data_dir
    _require_plan(data_dir)
    limit = SMOKE_LIMIT_PER_SPECIES if args.smoke else None
    refs = _split_refs(config, data_dir, limit)
    if not refs["train"] or not refs["val"]:
        raise DataError(
            "brak okien w train/val — uruchom `bird_recognizer prep cache`"
        )

    x_train, y_train, _ = features_from_refs(config, data_dir, refs["train"])
    x_val, y_val, groups_val = features_from_refs(config, data_dir, refs["val"])
    x_test, y_test, groups_test = features_from_refs(config, data_dir, refs["test"])
    if x_train.shape[0] > MAX_SVM_TRAIN_WINDOWS and args.clf == "svm":
        rng = np.random.default_rng(config.seed)
        keep = rng.choice(x_train.shape[0], MAX_SVM_TRAIN_WINDOWS, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]
        LOGGER.warning("SVM: ograniczam zbiór treningowy do %s okien", MAX_SVM_TRAIN_WINDOWS)

    started = time.perf_counter()
    model = MFCCBaseline.fit(x_train, y_train, kind=args.clf, seed=config.seed)
    duration = time.perf_counter() - started

    run = Run.create(config, model="mfcc", tag=args.clf, device="cpu")
    joblib.dump(model.model, run.path("model.joblib"))
    val_metrics = _summarise_predictions(
        model.scores(x_val), model.classes_, y_val, groups_val
    )
    test_metrics = (
        _summarise_predictions(
            model.scores(x_test), model.classes_, y_test, groups_test
        )
        if x_test.shape[0]
        else None
    )
    run.save_metrics(
        {
            "model": "mfcc",
            "clf": args.clf,
            "duration_s": round(duration, 2),
            "n_train_windows": int(x_train.shape[0]),
            "n_classes_seen": int(len(model.classes_)),
            "val": val_metrics,
            "test": test_metrics,
            "smoke": bool(args.smoke),
        }
    )
    row = base_index_row(
        run,
        config,
        pretrained="",
        aug="none",
        n_train=len(refs["train"]),
        n_val=len(refs["val"]),
        n_test=len(refs["test"]),
        val_macro_f1=val_metrics["recording_macro_f1"],
        test_macro_f1=test_metrics["recording_macro_f1"] if test_metrics else "",
        test_top1=test_metrics["recording_top1"] if test_metrics else "",
        test_top3=(
            test_metrics["recording_top3"]
            if test_metrics and test_metrics["recording_top3"] is not None
            else ""
        ),
        duration_s=round(duration, 2),
        notes=f"clf={args.clf}" + ("; smoke" if args.smoke else ""),
    )
    run.append_index(row)
    print(
        f"MFCC/{args.clf}: run={run.run_id} "
        f"(|train|={x_train.shape[0]} okien, {duration:.1f} s)"
    )
    print(
        f"  val: recording macro-F1={_fmt(val_metrics['recording_macro_f1'])} "
        f"top1={_fmt(val_metrics['recording_top1'])} "
        f"top3={_fmt(val_metrics['recording_top3'])}"
    )
    if test_metrics:
        print(
            f"  test: recording macro-F1={_fmt(test_metrics['recording_macro_f1'])} "
            f"top1={_fmt(test_metrics['recording_top1'])} "
            f"top3={_fmt(test_metrics['recording_top3'])}"
        )
    return 0


def _neural_training(
    config: Config,
    args: argparse.Namespace,
    *,
    finetune: bool = False,
) -> TrainingConfig:
    """Training budget for one neural run: transfer fine-tuning, smoke, ``--epochs``."""
    training = config.training
    if finetune:
        training = replace(training, lr=training.lr_finetune, epochs=training.epochs_finetune)
    if args.smoke:
        training = replace(training, epochs=1)
    elif args.epochs is not None:
        training = replace(training, epochs=args.epochs)
    return training


@dataclass
class NeuralRun:
    """Outcome of one training run, enough for CLI printing and ablation records."""

    run: Run
    parameters: int
    epochs_run: int
    duration_s: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None


def run_neural(
    config: Config,
    data_dir: Path,
    *,
    device: torch.device,
    model_name: str,
    tag: str = "",
    options: dict[str, object] | None = None,
    training: TrainingConfig | None = None,
    notes: str = "",
    smoke: bool = False,
    limit_per_species: int | None = None,
) -> NeuralRun:
    """Train one registered model end-to-end and write its run directory (FR-9)."""
    _require_plan(data_dir)
    refs = _split_refs(config, data_dir, limit_per_species)
    if not refs["train"] or not refs["val"]:
        raise DataError(
            "brak okien w train/val — uruchom `bird_recognizer prep cache`"
        )

    train_ds = WindowDataset(refs["train"], with_lengths=True)
    val_ds = WindowDataset(refs["val"], cache_size=8, with_lengths=True)
    test_ds = (
        WindowDataset(refs["test"], cache_size=8, with_lengths=True)
        if refs["test"]
        else None
    )
    run_config = replace(
        config, training=training if training is not None else config.training
    )
    # Model weights must be drawn after seeding, otherwise two runs with the same
    # config produce different initialisations (caught in the stage-4 gate).
    set_seed(config.seed)
    model = build_model(model_name, len(config.species), **(options or {})).to(device)
    run = Run.create(run_config, model=model_name, tag=tag, device=str(device))
    run.record_parameters(sum(parameter.numel() for parameter in model.parameters()))
    started = time.perf_counter()
    result = train_model(model, train_ds, val_ds, run_config, device, run=run)
    if result.best_state is not None:
        torch.save(result.best_state, run.path("best.pt"))
        model.load_state_dict(result.best_state)
    val_metrics = _model_metrics(model, val_ds, device, len(config.species))
    test_metrics = (
        _model_metrics(model, test_ds, device, len(config.species)) if test_ds else None
    )
    duration = time.perf_counter() - started
    run.save_metrics(
        {
            "model": model_name,
            "options": dict(options or {}),
            "parameters": result.parameters,
            "best_epoch": result.best_epoch,
            "best_val_window_macro_f1": round(result.best_val_macro_f1, 6),
            "stopped_early": result.stopped_early,
            "history": [record.as_dict() for record in result.history],
            "val": val_metrics,
            "test": test_metrics,
            "duration_s": round(duration, 2),
            "data_dir": str(data_dir),
            "limit_per_species": limit_per_species,
            "smoke": bool(smoke),
        }
    )
    row = base_index_row(
        run,
        run_config,
        pretrained="imagenet" if (options or {}).get("pretrained") else "none",
        pool=str((options or {}).get("pool", "")),
        aug=run_config.augment.preset,
        n_train=len(refs["train"]),
        n_val=len(refs["val"]),
        n_test=len(refs["test"]),
        val_macro_f1=val_metrics["recording_macro_f1"],
        test_macro_f1=test_metrics["recording_macro_f1"] if test_metrics else "",
        test_top1=test_metrics["recording_top1"] if test_metrics else "",
        test_top3=(
            test_metrics["recording_top3"]
            if test_metrics and test_metrics["recording_top3"] is not None
            else ""
        ),
        duration_s=round(duration, 2),
        notes="; ".join(part for part in (notes, "smoke" if smoke else "") if part),
    )
    run.append_index(row)
    return NeuralRun(
        run=run,
        parameters=result.parameters,
        epochs_run=len(result.history),
        duration_s=duration,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def _print_run(label: str, result: NeuralRun, device: torch.device) -> None:
    print(
        f"{label}: run={result.run.run_id} ({result.parameters} parametrów, "
        f"{result.epochs_run} epok, {result.duration_s:.1f} s, device={device})"
    )
    print(
        f"  val: recording macro-F1={_fmt(result.val_metrics['recording_macro_f1'])} "
        f"top1={_fmt(result.val_metrics['recording_top1'])} "
        f"top3={_fmt(result.val_metrics['recording_top3'])}"
    )
    if result.test_metrics:
        print(
            f"  test: recording macro-F1={_fmt(result.test_metrics['recording_macro_f1'])} "
            f"top1={_fmt(result.test_metrics['recording_top1'])} "
            f"top3={_fmt(result.test_metrics['recording_top3'])}"
        )


def _cmd_train_neural(
    config: Config,
    args: argparse.Namespace,
    *,
    model_name: str,
    tag: str = "",
    options: dict[str, object] | None = None,
    training: TrainingConfig | None = None,
    notes: str = "",
) -> int:
    """One neural run: cnn, crnn or transfer (SPEC FR-6/7/8, §10.5)."""
    device = _resolve_device(args)
    if getattr(args, "aug", None):
        config = replace(config, augment=replace(config.augment, preset=args.aug))
    limit = SMOKE_LIMIT_PER_SPECIES if args.smoke else None
    result = run_neural(
        config,
        config.paths.data_dir,
        device=device,
        model_name=model_name,
        tag=tag,
        options=options,
        training=training,
        notes=notes,
        smoke=bool(args.smoke),
        limit_per_species=limit,
    )
    _print_run(model_name.upper() + (f" ({tag})" if tag else ""), result, device)
    return 0


def _cmd_train_cnn(config: Config, args: argparse.Namespace) -> int:
    return _cmd_train_neural(
        config,
        args,
        model_name="cnn",
        training=_neural_training(config, args),
    )


def _cmd_train_crnn(config: Config, args: argparse.Namespace) -> int:
    options = {"pool": args.pool, "rnn": args.rnn, "n_mels": config.spectrogram.n_mels}
    return _cmd_train_neural(
        config,
        args,
        model_name="crnn",
        tag=args.pool,
        options=options,
        training=_neural_training(config, args),
        notes=f"rnn={args.rnn}",
    )


def _cmd_train_transfer(config: Config, args: argparse.Namespace) -> int:
    options = {"arch": args.arch, "pretrained": bool(args.pretrained)}
    discriminator = "pretrained" if args.pretrained else "scratch"
    return _cmd_train_neural(
        config,
        args,
        model_name="transfer",
        tag=f"{args.arch}-{discriminator}",
        options=options,
        training=_neural_training(config, args, finetune=bool(args.pretrained)),
        notes=f"arch={args.arch}",
    )


def _ablation_training(config: Config, args: argparse.Namespace) -> TrainingConfig:
    """One shared budget for every arm: same seed, split, epochs and lr (FR-11)."""
    training = config.training
    if args.smoke:
        return replace(training, epochs=1)
    if args.epochs is not None:
        return replace(training, epochs=args.epochs)
    return training


def _row_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def _cmd_ablate(config: Config, args: argparse.Namespace) -> int:
    """Run one ablation group: reference + variants, each as a separate run (FR-11)."""
    group = args.group
    ablation.validate_groups()
    device = _resolve_device(args)
    training = _ablation_training(config, args)
    limit = SMOKE_LIMIT_PER_SPECIES if args.smoke else None
    base_dir = config.paths.data_dir
    records: list[dict[str, object]] = []
    window_dirs: dict[str, str] = {}
    print(f"Ablacja [{group}]: {ablation.GROUP_TITLES[group]}")
    for variant in ablation.GROUPS[group]:
        run_config = ablation.config_for_variant(config, variant)
        data_dir = base_dir
        try:
            if variant.window_s != config.audio.window_s:
                data_dir = ablation.ensure_window_cache(config, variant.window_s, base_dir)
                window_dirs[f"{variant.window_s:g}s"] = str(data_dir)
            result = run_neural(
                run_config,
                data_dir,
                device=device,
                model_name=variant.model,
                tag=ablation.variant_tag(group, variant),
                options=variant.options,
                training=training,
                notes=f"{group}:{variant.label}",
                smoke=bool(args.smoke),
                limit_per_species=limit,
            )
        except (DataError, ModelError) as exc:
            records.append(
                ablation.run_record(
                    variant, group=group, status="missing", reason=str(exc)
                )
            )
            print(f"  UWAGA {variant.label}: pominięty — {exc}")
            continue
        test_f1 = (result.test_metrics or {}).get("recording_macro_f1")
        records.append(
            ablation.run_record(
                variant,
                group=group,
                status="ok",
                run_id=result.run.run_id,
                metrics={
                    "val_macro_f1": result.val_metrics["recording_macro_f1"],
                    "test_macro_f1": test_f1,
                },
            )
        )
        print(
            f"  {variant.label}: run={result.run.run_id} "
            f"test macro-F1={_fmt(test_f1)} ({result.duration_s:.1f} s)"
        )
    payload = ablation.payload(
        group,
        records,
        smoke=bool(args.smoke),
        epochs=training.epochs,
        device=str(device),
        window_data_dirs=window_dirs,
    )
    path = ablation.write_ablation(config.paths.runs_dir, payload)
    finished = sum(1 for record in records if record["status"] == "ok")
    print(f"Ablacja {group}: {finished}/{len(records)} runów; definicja -> {path}")
    return 0 if finished else 3


def _species_pairs(config: Config) -> dict[str, tuple[str, str]]:
    """Pairs declared in ``configs/species.yaml`` (``pair`` field); used by the report."""
    grouped: dict[str, list[str]] = {}
    for species in config.species:
        if species.pair:
            grouped.setdefault(species.pair, []).append(species.pl)
    return {
        name: (names[0], names[1])
        for name, names in grouped.items()
        if len(names) >= 2
    }


def _cmd_report(config: Config, args: argparse.Namespace) -> int:
    """Generate the Markdown results table from ``runs/index.csv`` (FR-14)."""
    rows = load_index(config.paths.runs_dir)
    if not rows:
        raise DataError(
            f"brak wpisów w {config.paths.runs_dir / 'index.csv'} — najpierw wytrenuj model"
        )
    selected = select_rows(rows, args.runs)
    path = write_report(
        args.out,
        selected,
        reference_id=selected[0]["run_id"],
        runs_dir=config.paths.runs_dir,
        species_pairs=_species_pairs(config),
    )
    print(f"Raport: {len(selected)} runów -> {path}")
    for row in selected:
        print(
            f"  {row['run_id']}: test macro-F1="
            f"{_fmt(_row_float(row, 'test_macro_f1'))}, "
            f"val macro-F1={_fmt(_row_float(row, 'val_macro_f1'))}"
        )
    return 0


def _cmd_eval(config: Config, args: argparse.Namespace) -> int:
    """Window and recording metrics for a saved run (FR-10)."""
    device = _resolve_device(args)
    bundle = evaluate_module.load_run(config, args.run, device=device)
    payload = evaluate_module.evaluate_split(bundle, args.split)
    recording = payload["recording"]
    vote = payload["vote"]
    print(
        f"Eval {bundle.run_id} / {args.split}: "
        f"nagrania={payload['n_recordings']}, okna={payload['n_windows']}"
    )
    print(
        f"  mean: macro-F1={recording['macro_f1']:.3f} "
        f"accuracy={recording['accuracy']:.3f} "
        f"balanced={recording['balanced_accuracy']:.3f} "
        f"top3={_fmt(recording['top3'])}"
    )
    print(
        f"  vote: macro-F1={vote['macro_f1']:.3f} accuracy={vote['accuracy']:.3f}"
    )
    print(f"  pliki: {bundle.directory / ('eval_' + args.split)}")
    return 0


def _cmd_aggregate(config: Config, args: argparse.Namespace) -> int:
    """Mean vs vote aggregation and the coverage/accuracy table (FR-10, §10.6)."""
    device = _resolve_device(args)
    bundle = evaluate_module.load_run(config, args.run, device=device)
    scores = evaluate_module.split_recording_scores(bundle, args.split)
    labels = list(range(len(bundle.config.species)))
    mean_predictions = scores.scores.argmax(axis=1)
    vote_order, vote_predictions = vote_by_group(scores.window_scores, scores.window_groups)
    truth_by_id = dict(zip(scores.order, scores.truth, strict=True))
    vote_truth = [truth_by_id[recording] for recording in vote_order]
    rows = [
        {
            "method": "mean",
            "selected": args.method == "mean",
            "n": int(len(scores.order)),
            "macro_f1": round(float(macro_f1(scores.truth, mean_predictions, labels=labels)), 6),
            "accuracy": round(float(accuracy(scores.truth, mean_predictions)), 6),
        },
        {
            "method": "vote",
            "selected": args.method == "vote",
            "n": int(len(vote_truth)),
            "macro_f1": round(float(macro_f1(vote_truth, vote_predictions, labels=labels)), 6),
            "accuracy": round(float(accuracy(vote_truth, vote_predictions)), 6),
        },
    ]
    coverage = coverage_curve(scores.truth, scores.scores)
    out_dir = bundle.directory / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(
        out_dir / "aggregation.csv", rows, ("method", "selected", "n", "macro_f1", "accuracy")
    )
    _write_rows(out_dir / "coverage.csv", coverage, ("threshold", "n", "coverage", "accuracy"))
    print(f"Agregacja {bundle.run_id} / {args.split}: n={len(scores.order)}")
    for row in rows:
        marker = "*" if row["selected"] else " "
        print(
            f" {marker} {row['method']}: macro-F1={row['macro_f1']:.3f} "
            f"accuracy={row['accuracy']:.3f}"
        )
    print("  coverage vs accuracy:")
    for row in coverage:
        print(
            f"    próg={row['threshold']:.2f}: n={row['n']} "
            f"coverage={row['coverage']:.2f} accuracy={_fmt(row['accuracy'])}"
        )
    return 0


def _write_rows(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _cmd_error_analysis(config: Config, args: argparse.Namespace) -> int:
    """Confusing pairs, per-species recall and worst recordings (FR-12)."""
    device = _resolve_device(args)
    bundle = evaluate_module.load_run(config, args.run, device=device)
    scores = evaluate_module.split_recording_scores(bundle, args.split)
    names = evaluate_module.species_names(bundle.config)
    catalog_file = catalog_path(bundle.data_dir)
    catalog = (
        {str(entry["id"]): entry for entry in load_catalog(catalog_file)}
        if catalog_file.is_file()
        else {}
    )
    predictions = [
        error_analysis_module.RecordingPrediction(
            recording_id=recording,
            truth=int(scores.truth[index]),
            predicted=int(scores.scores[index].argmax()),
            confidence=float(scores.scores[index].max()),
            p_true=float(scores.scores[index][scores.truth[index]]),
        )
        for index, recording in enumerate(scores.order)
    ]
    analysis = error_analysis_module.analyse(
        predictions,
        names=names,
        catalog=catalog,
        cache_root=bundle.data_dir / "cache",
        labels=list(range(len(bundle.config.species))),
        top=args.top,
    )
    paths = error_analysis_module.write_analysis(bundle.directory / "analysis", analysis)
    print(
        f"Analiza błędów: {bundle.run_id} / {args.split}, "
        f"nagrania={analysis['n']}"
    )
    pairs = ", ".join(
        f"{row['truth_name']}→{row['predicted_name']} ({row['count']})"
        for row in analysis["pairs"][:5]
    )
    print(f"  pary: {pairs or 'brak pomyłek'}")
    print(
        f"  podejrzani o szum etykiet (also≠puste) w najgorszych: "
        f"{analysis['label_noise_suspects']}/{len(analysis['worst'])}"
    )
    print(f"  pliki: {paths['markdown']}")
    print(f"         {paths['pairs']}, {paths['per_species']}, {paths['worst']}")
    return 0


def _cmd_demo(config: Config, args: argparse.Namespace) -> int:
    """Gradio demo; ``--smoke`` checks the path without starting a server (FR-13)."""
    from . import demo as demo_module

    device = _resolve_device(args)
    bundle = evaluate_module.load_run(config, args.run, device=device)
    if args.smoke:
        return demo_module.smoke_check(bundle)
    print(f"Demo: run {bundle.run_id}, model {bundle.model_name}, device {device}")
    print(f"  otwieram http://127.0.0.1:{args.port} (Ctrl+C kończy)")
    demo_module.launch(bundle, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; maps :class:`AppError` to the exit codes from SPEC §8."""
    configure_console()
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, env_file=args.env_file)
        config = _apply_overrides(config, args)
        key = get_api_key()
        setup_logging("DEBUG" if args.verbose else "INFO", secrets=[key] if key else [])
        if args.command == "data":
            if args.data_command == "catalog":
                return _cmd_catalog(config, args)
            if args.data_command == "download":
                return _cmd_download(config, args)
            if args.data_command == "split":
                return _cmd_split(config, args)
        if args.command == "prep" and args.prep_command == "cache":
            return _cmd_cache(config, args)
        if args.command == "ablate":
            return _cmd_ablate(config, args)
        if args.command == "report":
            return _cmd_report(config, args)
        if args.command == "eval":
            return _cmd_eval(config, args)
        if args.command == "aggregate":
            return _cmd_aggregate(config, args)
        if args.command == "error-analysis":
            return _cmd_error_analysis(config, args)
        if args.command == "demo":
            return _cmd_demo(config, args)
        if args.command == "train":
            if args.train_command == "mfcc":
                return _cmd_train_mfcc(config, args)
            if args.train_command == "cnn":
                return _cmd_train_cnn(config, args)
            if args.train_command == "crnn":
                return _cmd_train_crnn(config, args)
            if args.train_command == "transfer":
                return _cmd_train_transfer(config, args)
        LOGGER.error("Nieznana komenda: %s", args.command)
        return 1
    except AppError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("Przerwano.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
