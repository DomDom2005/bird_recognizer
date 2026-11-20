"""Single-variable ablation matrix (SPEC FR-11, §11.1).

Each group has one **reference** variant (the first one) and variants that differ from
it in exactly one experimental knob: ``architecture`` (which bundles the model choice:
cnn / crnn-rnn-pool / arch-weights), ``window_s`` or ``aug``. The JSON written to
``runs/ablation_<group>.json`` records every run id together with that single
difference, so the claim "one change = one run" is checkable without reading logs.

The window group needs a cache built for a different window geometry; the files under
``data/window5`` and ``data/window10`` are hard links to ``data/raw`` (no extra audio
copy) plus the same catalog and split.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config
from .data.dataset import build_cache
from .errors import DataError

LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOW_S = 3.0
#: SPEC §10.1: hop is 50% of the window, so a longer window also has a longer hop.
WINDOW_HOP_RATIO = 0.5


@dataclass(frozen=True)
class Variant:
    """One level of one ablation factor."""

    label: str
    model: str
    options: dict[str, object] = field(default_factory=dict)
    window_s: float = DEFAULT_WINDOW_S
    aug: str = "basic"
    description: str = ""

    def architecture(self) -> str:
        """Model recipe as one string — the ablation knob for model-level groups."""
        if self.model == "cnn":
            return "cnn"
        if self.model == "crnn":
            rnn = self.options.get("rnn", "gru")
            pool = self.options.get("pool", "mean")
            return f"crnn-{rnn}-{pool}"
        if self.model == "transfer":
            weights = "imagenet" if self.options.get("pretrained") else "scratch"
            return f"{self.options.get('arch', 'resnet18')}-{weights}"
        return self.model

    def settings(self) -> dict[str, Any]:
        """The three knobs an ablation may vary; everything else must stay equal."""
        return {
            "architecture": self.architecture(),
            "window_s": float(self.window_s),
            "aug": self.aug,
        }

    def tag(self, group: str) -> str:
        return f"{group}-{self.label}"


#: First variant of each group is its reference; the rest differ in exactly one knob.
GROUPS: dict[str, tuple[Variant, ...]] = {
    "rnn": (
        Variant("cnn", "cnn", description="odniesienie: CNN bez rekurencji"),
        Variant(
            "crnn-gru",
            "crnn",
            {"pool": "mean", "rnn": "gru"},
            description="CRNN z BiGRU(128) i poolingiem mean",
        ),
        Variant(
            "crnn-lstm",
            "crnn",
            {"pool": "mean", "rnn": "lstm"},
            description="CRNN z BiLSTM(128) i poolingiem mean",
        ),
    ),
    "window": (
        Variant("3s", "cnn", window_s=3.0, description="domyślne okno 3 s"),
        Variant("5s", "cnn", window_s=5.0, description="okno 5 s (hop 2,5 s)"),
        Variant("10s", "cnn", window_s=10.0, description="okno 10 s (hop 5 s)"),
    ),
    "aug": (
        Variant("basic", "cnn", aug="basic", description="przesunięcie + szum"),
        Variant("none", "cnn", aug="none", description="bez augmentacji"),
        Variant("full", "cnn", aug="full", description="+ SpecAugment i mixup"),
    ),
    "pretrained": (
        Variant(
            "scratch",
            "transfer",
            {"arch": "resnet18", "pretrained": False},
            description="ResNet18 od zera",
        ),
        Variant(
            "imagenet",
            "transfer",
            {"arch": "resnet18", "pretrained": True},
            description="ResNet18 z wagami ImageNet",
        ),
    ),
    "pool": (
        Variant(
            "mean",
            "crnn",
            {"pool": "mean", "rnn": "gru"},
            description="CRNN z poolingiem mean",
        ),
        Variant(
            "attention",
            "crnn",
            {"pool": "attention", "rnn": "gru"},
            description="CRNN z poolingiem attention",
        ),
    ),
}

GROUP_TITLES: dict[str, str] = {
    "rnn": "Czy rekurencja daje przewagę (CNN vs CRNN-GRU vs CRNN-LSTM)",
    "window": "Ile kontekstu potrzebuje model (3 s vs 5 s vs 10 s)",
    "aug": "Co daje augmentacja (none vs basic vs full)",
    "pretrained": "Ile daje transfer z ImageNet (od zera vs wstępnie wytrenowane)",
    "pool": "Czy attention nad sekwencją pomaga (mean vs attention)",
}


def variant_tag(group: str, variant: Variant) -> str:
    return variant.tag(group)


def single_difference(reference: Variant, variant: Variant) -> dict[str, tuple[Any, Any]]:
    """Keys on which ``variant`` differs from ``reference`` (empty for the reference)."""
    left, right = reference.settings(), variant.settings()
    return {key: (left[key], right[key]) for key in left if left[key] != right[key]}


def definition_violations(group: str) -> list[str]:
    """Every non-reference variant must differ from the reference in exactly one knob."""
    if group not in GROUPS:
        raise DataError(f"nieznana grupa ablacji '{group}'; dostępne: {sorted(GROUPS)}")
    reference = GROUPS[group][0]
    violations: list[str] = []
    for variant in GROUPS[group][1:]:
        diff = single_difference(reference, variant)
        if len(diff) != 1:
            violations.append(f"{group}/{variant.label}: {len(diff)} różnic {diff}")
    return violations


def validate_groups() -> None:
    problems = {group: definition_violations(group) for group in GROUPS}
    broken = {group: items for group, items in problems.items() if items}
    if broken:
        raise DataError(f"definicje ablacji łamią zasadę jednej zmiany: {broken}")


# --------------------------------------------------------------------------- #
# Window variants: separate cache, same audio (hard links, no second copy)
# --------------------------------------------------------------------------- #
def window_data_dir(base_data_dir: Path, window_s: float) -> Path:
    return Path(base_data_dir) / f"window{int(round(window_s))}"


def config_for_window(config: Config, window_s: float) -> Config:
    hop_s = window_s * WINDOW_HOP_RATIO
    return replace(config, audio=replace(config.audio, window_s=window_s, hop_s=hop_s))


def config_for_variant(config: Config, variant: Variant) -> Config:
    return replace(
        config_for_window(config, variant.window_s),
        augment=replace(config.augment, preset=variant.aug),
    )


def prepare_window_data(base_data_dir: Path, window_s: float) -> Path:
    """Create ``data/window<N>`` with hard-linked audio and copied catalog/splits."""
    base = Path(base_data_dir)
    target = window_data_dir(base, window_s)
    (target / "raw").mkdir(parents=True, exist_ok=True)
    for source in sorted((base / "raw").iterdir()):
        if not source.is_file():
            continue
        destination = target / "raw" / source.name
        if destination.exists():
            continue
        try:
            os.link(source, destination)  # same volume: no extra bytes
        except OSError:
            shutil.copy2(source, destination)
    for name in ("catalog.jsonl", "splits.json"):
        source = base / name
        destination = target / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    return target


def ensure_window_cache(config: Config, window_s: float, base_data_dir: Path) -> Path:
    """Prepare the window data dir and (re)build its cache; idempotent."""
    target = prepare_window_data(base_data_dir, window_s)
    report = build_cache(config_for_window(config, window_s), target)
    LOGGER.info(
        "cache okna %ss: przetworzone %s, pominięte %s, błędy %s, okna %s/%s",
        window_s,
        report.processed,
        report.skipped,
        len(report.errors),
        report.windows_kept,
        report.windows_total,
    )
    return target


# --------------------------------------------------------------------------- #
# Result JSON
# --------------------------------------------------------------------------- #
def ablation_path(runs_dir: Path, group: str) -> Path:
    return Path(runs_dir) / f"ablation_{group}.json"


def run_record(
    variant: Variant,
    *,
    group: str,
    status: str,
    run_id: str | None = None,
    reason: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One entry of ``ablation_<group>.json``; ``diff`` proves the single change."""
    reference = GROUPS[group][0]
    diff = single_difference(reference, variant)
    payload: dict[str, Any] = {
        "label": variant.label,
        "description": variant.description,
        "status": status,
        "run_id": run_id,
        "reason": reason,
        "settings": variant.settings(),
        "diff": {key: list(values) for key, values in diff.items()},
    }
    if metrics:
        payload["val_macro_f1"] = metrics.get("val_macro_f1")
        payload["test_macro_f1"] = metrics.get("test_macro_f1")
    return payload


def payload(
    group: str,
    records: list[dict[str, Any]],
    *,
    smoke: bool,
    epochs: int,
    device: str,
    window_data_dirs: dict[str, str] | None = None,
) -> dict[str, Any]:
    reference = GROUPS[group][0]
    return {
        "group": group,
        "title": GROUP_TITLES[group],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "smoke": bool(smoke),
        "epochs_budget": int(epochs),
        "device": device,
        "note": (
            "Wszystkie warianty dzielą seed, split i budżet epok; jedyną różnicą jest "
            "pole z `diff`. W grupie `pretrained` obie gałęzie uczą się z tym samym lr "
            "(1e-3), bo zmiana lr byłaby drugą zmianą."
        ),
        "reference": {"label": reference.label, "settings": reference.settings()},
        "runs": records,
        "window_data_dirs": window_data_dirs or {},
    }


def write_ablation(runs_dir: Path, data: dict[str, Any]) -> Path:
    path = ablation_path(runs_dir, str(data["group"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_ablation(path: Path) -> dict[str, Any]:
    payload_data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload_data, dict):
        raise DataError(f"{path}: oczekiwano obiektu JSON")
    return payload_data
