"""Configuration loading and validation (SPEC §3, §10, §13).

Single source of truth for thresholds and hyper-parameters is
``configs/default.yaml``; the species list lives in ``configs/species.yaml`` and
its order defines class indices. Nothing is read from globals: ``load_config()``
returns a frozen :class:`Config` and callers pass it explicitly (NFR-5).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from .errors import ConfigError, MissingApiKeyError

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_SPECIES_FILENAME = "species.yaml"

#: Accepted API-key variables in priority order. The lowercase alias is what the
#: owner's existing `.env` uses; ``BIRD_XC_API_KEY`` is the documented name.
API_KEY_ENV_NAMES: tuple[str, ...] = ("BIRD_XC_API_KEY", "xc_api_key")

_API_HOST = "xeno-canto.org"
_LATIN_NAME_RE = re.compile(r"^[A-Z][a-z]+ [a-z]+$")
_QUALITY_RE = re.compile(r"[<>]?[A-E]")
_STATUSES = ("identified", "unidentified", "questioned", "unconfirmed")
_PROTECTED_SPECIES_KEYS = {"pl", "latin", "pair"}


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ApiConfig:
    """How to talk to xeno-canto API v3 (SPEC §4, §9)."""

    base_url: str = "https://xeno-canto.org/api/3/recordings"
    country: str = "poland"
    quality: str = '">C"'
    #: Empty string = tag omitted. Live probe 2026-11-02: ``status:identified``
    #: returns HTTP 400 "only accepts queries using tags" (docs/api-notes.md §3).
    status: str = ""
    per_page: int = 100
    request_interval_s: float = 1.0
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0
    download_timeout_s: float = 300.0
    max_retries: int = 5


@dataclass(frozen=True)
class AudioConfig:
    """Audio front-end (SPEC §10.1)."""

    sample_rate: int = 32000
    highpass_hz: float = 100.0
    highpass_order: int = 4
    window_s: float = 3.0
    hop_s: float = 1.5


@dataclass(frozen=True)
class SpectrogramConfig:
    """log-mel parameters (SPEC §10.1)."""

    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 150.0
    fmax: float = 15000.0
    power: float = 2.0
    top_db: float = 80.0
    normalise: bool = True


@dataclass(frozen=True)
class GatingConfig:
    """Silence gating thresholds (SPEC §10.2)."""

    abs_floor: float = 1e-3
    rel_floor: float = 0.1
    percentile: float = 90.0


@dataclass(frozen=True)
class AugmentConfig:
    """Augmentation parameters; ``preset`` selects none/basic/full (SPEC §10.3)."""

    preset: str = "basic"
    shift_s: float = 1.0
    gain_db: float = 3.0
    noise_snr_min: float = 5.0
    noise_snr_max: float = 20.0
    mixup_alpha: float = 0.4
    freq_masks: int = 2
    freq_mask_width: int = 16
    time_masks: int = 2
    time_mask_width: int = 20


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop defaults (SPEC §10.5)."""

    batch_size: int = 32
    optimizer: str = "adamw"
    lr: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    epochs_finetune: int = 20
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3
    grad_clip: float = 5.0
    class_weight: str = "balanced"
    label_smoothing: float = 0.0
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    num_workers: int = 0


@dataclass(frozen=True)
class PathsConfig:
    """Where data and run artefacts live; both are outside the repo (SPEC §7)."""

    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")


@dataclass(frozen=True)
class Species:
    """One class of the classifier (SPEC §3, D1)."""

    pl: str
    latin: str
    pair: str | None = None


@dataclass(frozen=True)
class Config:
    """Validated configuration passed explicitly to every component."""

    species: tuple[Species, ...]
    api: ApiConfig = field(default_factory=ApiConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 42
    config_path: Path | None = None

    def species_labels(self) -> dict[str, int]:
        """Map polish species name to class index (order of ``species.yaml``)."""
        return {s.pl: i for i, s in enumerate(self.species)}


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Brak pliku konfiguracji: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Nie mogę odczytać {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: niepoprawny YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"{path}: plik jest pusty")
    return data


def _coerce(value: Any, expected: str, key: str, where: str) -> Any:
    if expected == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: '{key}' musi być liczbą całkowitą")
        return value
    if expected == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: '{key}' musi być liczbą")
        return float(value)
    if expected == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: '{key}' musi być true/false")
        return value
    if expected == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{where}: '{key}' musi być łańcuchem")
        return value
    if expected == "Path":
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ConfigError(f"{where}: '{key}' musi być ścieżką")
        return Path(value)
    raise ConfigError(f"{where}: nieobsługiwany typ pola '{key}' ({expected})")


def _build(cls: type, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: oczekiwano mapy klucz → wartość")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{where}: nieznane klucze {unknown}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], f.type, f.name, f"{where}.{f.name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{where}: brakuje klucza '{f.name}'")
    return cls(**kwargs)


def _load_species(path: Path) -> tuple[Species, ...]:
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or "species" not in raw:
        raise ConfigError(f"{path}: oczekiwano klucza 'species'")
    unknown = sorted(set(raw) - {"species"})
    if unknown:
        raise ConfigError(f"{path}: nieznane klucze {unknown}")
    items = raw["species"]
    if not isinstance(items, list) or not items:
        raise ConfigError(f"{path}: 'species' musi być niepustą listą")

    species: list[Species] = []
    seen_latin: set[str] = set()
    for index, item in enumerate(items):
        where = f"{path}: species[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where}: oczekiwano mapy")
        unknown = sorted(set(item) - _PROTECTED_SPECIES_KEYS)
        if unknown:
            raise ConfigError(f"{where}: nieznane klucze {unknown}")
        pl = item.get("pl")
        latin = item.get("latin")
        pair = item.get("pair")
        if not isinstance(pl, str) or not pl.strip():
            raise ConfigError(f"{where}: 'pl' musi być niepustym łańcuchem")
        if not isinstance(latin, str) or not _LATIN_NAME_RE.match(latin):
            raise ConfigError(
                f"{where}: 'latin' musi być nazwą dwuczłonową, np. 'Parus major'"
            )
        if pair is not None and (not isinstance(pair, str) or not pair.strip()):
            raise ConfigError(f"{where}: 'pair' musi być łańcuchem albo pominięte")
        if latin in seen_latin:
            raise ConfigError(f"{where}: gatunek '{latin}' występuje dwa razy")
        seen_latin.add(latin)
        species.append(Species(pl=pl.strip(), latin=latin, pair=pair))
    return tuple(species)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_api(api: ApiConfig) -> None:
    parts = urlsplit(api.base_url)
    if parts.scheme != "https" or parts.netloc != _API_HOST:
        raise ConfigError(
            f"api.base_url musi wskazywać https://{_API_HOST}/api/3/recordings "
            "(tylko oficjalne API — SPEC §0 pkt 7)"
        )
    if not api.country.strip():
        raise ConfigError("api.country musi być niepuste (tag cnt:<kraj>)")
    if not 50 <= api.per_page <= 500:
        raise ConfigError("api.per_page musi być w zakresie 50..500 (SPEC §4.1)")
    if api.request_interval_s < 0:
        raise ConfigError("api.request_interval_s nie może być ujemne")
    if api.max_retries < 0:
        raise ConfigError("api.max_retries nie może być ujemny")
    if api.status and api.status not in _STATUSES:
        raise ConfigError(f"api.status musi być pusty albo jednym z {_STATUSES}")
    quality = api.quality.strip()
    if len(quality) >= 2 and quality[0] == quality[-1] == '"':
        quality = quality[1:-1]
    if not _QUALITY_RE.fullmatch(quality):
        raise ConfigError('api.quality: oczekiwano skali xeno-canto, np. q:">C" (A/B)')


def _validate_audio(audio: AudioConfig, spectrogram: SpectrogramConfig) -> None:
    if not 8000 <= audio.sample_rate <= 96000:
        raise ConfigError("audio.sample_rate musi być w zakresie 8000..96000")
    if audio.window_s <= 0 or audio.hop_s <= 0:
        raise ConfigError("audio.window_s i audio.hop_s muszą być dodatnie")
    if audio.hop_s > audio.window_s:
        raise ConfigError("audio.hop_s nie może przekraczać audio.window_s")
    if not 0 <= audio.highpass_hz < audio.sample_rate / 2:
        raise ConfigError("audio.highpass_hz musi być poniżej częstotliwości Nyquista")
    if spectrogram.n_fft <= 0 or spectrogram.hop_length <= 0 or spectrogram.n_mels <= 0:
        raise ConfigError("spectrogram: n_fft, hop_length i n_mels muszą być dodatnie")
    if spectrogram.n_fft < spectrogram.win_length:
        raise ConfigError("spectrogram.n_fft nie może być mniejsze niż win_length")
    if spectrogram.fmin < 0 or spectrogram.fmax <= spectrogram.fmin:
        raise ConfigError("spectrogram: wymagane 0 <= fmin < fmax")
    if spectrogram.fmax >= audio.sample_rate / 2:
        raise ConfigError("spectrogram.fmax musi być poniżej częstotliwości Nyquista")
    if spectrogram.power <= 0 or spectrogram.top_db <= 0:
        raise ConfigError("spectrogram.power i spectrogram.top_db muszą być dodatnie")


def _validate_gating(gating: GatingConfig) -> None:
    if not 0 <= gating.abs_floor < 1:
        raise ConfigError("gating.abs_floor musi być w zakresie [0, 1)")
    if not 0 <= gating.rel_floor < 1:
        raise ConfigError("gating.rel_floor musi być w zakresie [0, 1)")
    if not 0 < gating.percentile <= 100:
        raise ConfigError("gating.percentile musi być w zakresie (0, 100]")


def _validate_augment(augment: AugmentConfig) -> None:
    if augment.preset not in {"none", "basic", "full"}:
        raise ConfigError("augment.preset musi być jednym z: none, basic, full")
    if augment.shift_s < 0 or augment.gain_db < 0:
        raise ConfigError("augment.shift_s i augment.gain_db nie mogą być ujemne")
    if not 0 < augment.noise_snr_min <= augment.noise_snr_max:
        raise ConfigError("augment: wymagane 0 < noise_snr_min <= noise_snr_max")
    if not 0 <= augment.mixup_alpha <= 1:
        raise ConfigError("augment.mixup_alpha musi być w zakresie [0, 1]")
    for name in ("freq_masks", "freq_mask_width", "time_masks", "time_mask_width"):
        if getattr(augment, name) < 0:
            raise ConfigError(f"augment.{name} nie może być ujemne")


def _validate_training(training: TrainingConfig) -> None:
    if training.batch_size <= 0 or training.epochs <= 0:
        raise ConfigError("training.batch_size i training.epochs muszą być dodatnie")
    if training.lr <= 0 or training.lr_finetune <= 0 or training.grad_clip <= 0:
        raise ConfigError("training: lr, lr_finetune i grad_clip muszą być dodatnie")
    if training.weight_decay < 0 or training.early_stopping_min_delta < 0:
        raise ConfigError("training: weight_decay i early_stopping_min_delta >= 0")
    if not 0 <= training.label_smoothing <= 1:
        raise ConfigError("training.label_smoothing musi być w zakresie [0, 1]")
    if training.early_stopping_patience < 0 or training.num_workers < 0:
        raise ConfigError("training: patience i num_workers nie mogą być ujemne")
    if training.optimizer not in {"adamw", "sgd"}:
        raise ConfigError("training.optimizer musi być adamw albo sgd")
    if training.scheduler not in {"cosine", "none"}:
        raise ConfigError("training.scheduler musi być cosine albo none")
    if training.class_weight not in {"balanced", "none"}:
        raise ConfigError("training.class_weight musi być balanced albo none")


def _validate_config(config: Config) -> None:
    if len(config.species) < 2:
        raise ConfigError("Potrzebne są co najmniej 2 gatunki (klasyfikator)")
    if config.seed < 0:
        raise ConfigError("seed nie może być ujemny")
    _validate_api(config.api)
    _validate_audio(config.audio, config.spectrogram)
    _validate_gating(config.gating)
    _validate_augment(config.augment)
    _validate_training(config.training)
    for path in (config.paths.data_dir, config.paths.runs_dir):
        if not str(path).strip():
            raise ConfigError("paths: data_dir i runs_dir muszą być niepuste")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_env_file(path: Path | str | None = ".env", *, override: bool = False) -> None:
    """Load ``.env`` if it exists; never fails when the file is absent."""
    if path is None:
        return
    env_path = Path(path)
    if env_path.exists():
        load_dotenv(env_path, override=override)


def get_api_key(
    environ: Mapping[str, str] | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Return the xeno-canto API key, never logging it (SPEC §13)."""
    env = os.environ if environ is None else environ
    for name in API_KEY_ENV_NAMES:
        value = (env.get(name) or "").strip()
        if value:
            return value
    if required:
        raise MissingApiKeyError(
            "Brak klucza API xeno-canto. Ustaw BIRD_XC_API_KEY w `.env` "
            "(wzór: .env.example). Klucz jest osobisty i nie może trafić do repo."
        )
    return None


def load_config(
    config_path: Path | str | None = None,
    *,
    species_path: Path | str | None = None,
    env_file: Path | str | None = ".env",
) -> Config:
    """Load, cross-check and validate configuration (SPEC §10)."""
    if env_file is not None:
        load_env_file(env_file)

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: oczekiwano mapy klucz → wartość")
    allowed = {"seed", "paths", "api", "audio", "spectrogram", "gating", "augment", "training"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{path}: nieznane sekcje {unknown}")

    seed = _coerce(raw.get("seed", 42), "int", "seed", str(path))
    api = _build(ApiConfig, raw.get("api"), f"{path}:api")
    audio = _build(AudioConfig, raw.get("audio"), f"{path}:audio")
    spectrogram = _build(SpectrogramConfig, raw.get("spectrogram"), f"{path}:spectrogram")
    gating = _build(GatingConfig, raw.get("gating"), f"{path}:gating")
    augment = _build(AugmentConfig, raw.get("augment"), f"{path}:augment")
    training = _build(TrainingConfig, raw.get("training"), f"{path}:training")
    paths = _build(PathsConfig, raw.get("paths"), f"{path}:paths")

    species_path = (
        Path(species_path) if species_path is not None else path.parent / DEFAULT_SPECIES_FILENAME
    )
    species = _load_species(species_path)

    config = Config(
        species=species,
        api=api,
        audio=audio,
        spectrogram=spectrogram,
        gating=gating,
        augment=augment,
        training=training,
        paths=paths,
        seed=seed,
        config_path=path,
    )
    _validate_config(config)
    return config
