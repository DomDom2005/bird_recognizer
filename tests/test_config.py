"""Configuration tests (SPEC §12, §3, §10)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from bird_recognizer.config import get_api_key, load_config
from bird_recognizer.errors import ConfigError, MissingApiKeyError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
SPECIES_PATH = REPO_ROOT / "configs" / "species.yaml"


def _load_raw() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    raw = _load_raw()
    mutate(raw)
    target = tmp_dir / "default.yaml"
    target.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return target


def _load(tmp_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> Any:
    return load_config(_write_config(tmp_dir, mutate), species_path=SPECIES_PATH, env_file=None)


# --------------------------------------------------------------------------- #
# Shipped configuration
# --------------------------------------------------------------------------- #
def test_shipped_config_loads(config: Any) -> None:
    assert config.config_path == CONFIG_PATH
    assert config.species


def test_species_match_decision_d1(config: Any) -> None:
    assert len(config.species) == 12
    latin = [s.latin for s in config.species]
    assert len(set(latin)) == 12
    pairs = {s.pair for s in config.species if s.pair}
    assert pairs == {
        "wrobel-mazurek",
        "bogatka-modra",
        "kos-kwiczol",
        "pierwiosnek-piecuszek",
    }
    paired = [s for s in config.species if s.pair]
    assert len(paired) == 8


def test_species_order_defines_labels(config: Any) -> None:
    labels = config.species_labels()
    assert labels["wróbel"] == 0
    assert labels["dzwoniec"] == 11


def test_defaults_match_spec(config: Any) -> None:
    assert config.seed == 42
    assert config.audio.sample_rate == 32000
    assert config.audio.window_s == 3.0
    assert config.audio.hop_s == 1.5
    assert config.spectrogram.n_mels == 128
    assert config.spectrogram.hop_length == 512
    assert config.spectrogram.fmax == 15000
    assert config.api.per_page == 100
    assert config.api.country == "poland"
    assert config.training.batch_size == 32
    assert str(config.paths.data_dir) == "data"


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_unknown_top_level_section_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["bogus"] = 1

    with pytest.raises(ConfigError, match="nieznane sekcje"):
        _load(tmp_dir, mutate)


def test_unknown_key_inside_section_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["audio"]["bogus"] = 1

    with pytest.raises(ConfigError, match="nieznane klucze"):
        _load(tmp_dir, mutate)


def test_wrong_type_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["audio"]["sample_rate"] = "32000"

    with pytest.raises(ConfigError, match="liczbą całkowitą"):
        _load(tmp_dir, mutate)


def test_hop_longer_than_window_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["audio"]["hop_s"] = 5.0

    with pytest.raises(ConfigError, match="hop_s"):
        _load(tmp_dir, mutate)


def test_fmax_above_nyquist_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["spectrogram"]["fmax"] = 16000

    with pytest.raises(ConfigError, match="Nyquista"):
        _load(tmp_dir, mutate)


def test_per_page_out_of_range_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["per_page"] = 10

    with pytest.raises(ConfigError, match="per_page"):
        _load(tmp_dir, mutate)


def test_foreign_api_host_is_rejected(tmp_dir: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["api"]["base_url"] = "https://example.org/api/3/recordings"

    with pytest.raises(ConfigError, match="xeno-canto"):
        _load(tmp_dir, mutate)


def test_missing_species_file_is_rejected(tmp_dir: Path) -> None:
    with pytest.raises(ConfigError, match="Brak pliku"):
        load_config(CONFIG_PATH, species_path=tmp_dir / "missing.yaml", env_file=None)


def test_species_must_have_two_part_latin_name(tmp_dir: Path) -> None:
    species_file = tmp_dir / "species.yaml"
    species_file.write_text(
        yaml.safe_dump(
            {
                "species": [
                    {"pl": "wróbel", "latin": "Passer"},
                    {"pl": "mazurek", "latin": "Passer montanus"},
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="dwuczłonową"):
        load_config(CONFIG_PATH, species_path=species_file, env_file=None)


# --------------------------------------------------------------------------- #
# API key handling (SPEC §13)
# --------------------------------------------------------------------------- #
def test_api_key_from_documented_name() -> None:
    assert get_api_key({"BIRD_XC_API_KEY": "doc-key"}) == "doc-key"


def test_api_key_from_legacy_alias() -> None:
    assert get_api_key({"xc_api_key": "alias-key"}) == "alias-key"


def test_documented_name_wins_over_alias() -> None:
    environ = {"BIRD_XC_API_KEY": "doc-key", "xc_api_key": "alias-key"}
    assert get_api_key(environ) == "doc-key"


def test_missing_key_returns_none_when_optional() -> None:
    assert get_api_key({}) is None


def test_missing_key_is_config_error_with_exit_code_2() -> None:
    with pytest.raises(MissingApiKeyError) as excinfo:
        get_api_key({}, required=True)
    assert excinfo.value.exit_code == 2
    assert "BIRD_XC_API_KEY" in str(excinfo.value)


def test_missing_env_file_is_ignored(tmp_dir: Path) -> None:
    config = load_config(CONFIG_PATH, env_file=tmp_dir / "absent.env")
    assert config.species
