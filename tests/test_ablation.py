"""Ablation matrix and report tests (SPEC FR-11, FR-14, §11.1-11.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_recognizer import ablation
from bird_recognizer.cli import main
from bird_recognizer.errors import DataError
from bird_recognizer.eval.report import render_report, render_table, select_rows, write_report
from bird_recognizer.train.runner import load_index
from support import make_catalog_with_audio

GROUP_NAMES = ("rnn", "window", "aug", "pretrained", "pool")


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", GROUP_NAMES)
def test_each_group_has_reference_and_single_difference(group: str) -> None:
    variants = ablation.GROUPS[group]
    assert len(variants) >= 2
    assert ablation.definition_violations(group) == []
    reference = variants[0]
    assert ablation.single_difference(reference, reference) == {}
    for variant in variants[1:]:
        assert len(ablation.single_difference(reference, variant)) == 1


def test_every_group_has_a_title() -> None:
    assert set(ablation.GROUP_TITLES) == set(ablation.GROUPS)
    assert all(ablation.GROUP_TITLES[group] for group in ablation.GROUPS)


def test_variant_tags_are_unique() -> None:
    for group, variants in ablation.GROUPS.items():
        tags = [ablation.variant_tag(group, variant) for variant in variants]
        assert len(tags) == len(set(tags))


def test_reference_of_rnn_group_is_the_plain_cnn() -> None:
    reference = ablation.GROUPS["rnn"][0]
    assert reference.model == "cnn"
    assert reference.architecture() == "cnn"


def test_pool_group_differs_only_in_pooling() -> None:
    reference, attention = ablation.GROUPS["pool"]
    diff = ablation.single_difference(reference, attention)
    assert diff == {"architecture": ("crnn-gru-mean", "crnn-gru-attention")}


def test_unknown_group_is_rejected() -> None:
    with pytest.raises(DataError, match="nieznana grupa"):
        ablation.definition_violations("window_size")


# --------------------------------------------------------------------------- #
# Variant configuration and window data
# --------------------------------------------------------------------------- #
def test_config_for_variant_sets_window_and_augmentation(config) -> None:
    variant = ablation.Variant("5s", "cnn", window_s=5.0, aug="full")
    applied = ablation.config_for_variant(config, variant)
    assert applied.audio.window_s == 5.0
    assert applied.audio.hop_s == 2.5
    assert applied.augment.preset == "full"
    # the source config must stay untouched
    assert config.audio.window_s == 3.0


def test_window_data_dir_name() -> None:
    assert ablation.window_data_dir(Path("data"), 5.0).name == "window5"


def test_prepare_window_data_links_raw_and_copies_split(tmp_dir: Path) -> None:
    base = tmp_dir / "data"
    (base / "raw").mkdir(parents=True)
    (base / "raw" / "XC1.mp3").write_bytes(b"audio")
    (base / "catalog.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "splits.json").write_text('{"splits": {}}', encoding="utf-8")

    target = ablation.prepare_window_data(base, 5.0)
    linked = target / "raw" / "XC1.mp3"
    assert linked.read_bytes() == b"audio"
    assert linked.stat().st_nlink >= 2, "oczekiwano hard linku, nie kopii audio"
    assert (target / "catalog.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert (target / "splits.json").is_file()


# --------------------------------------------------------------------------- #
# JSON payload
# --------------------------------------------------------------------------- #
def test_run_record_reports_the_single_difference() -> None:
    variant = ablation.GROUPS["rnn"][1]
    record = ablation.run_record(variant, group="rnn", status="ok", run_id="r1")
    assert record["diff"] == {"architecture": ["cnn", "crnn-gru-mean"]}
    assert record["status"] == "ok"


def test_payload_roundtrip(tmp_dir: Path) -> None:
    records = [
        ablation.run_record(
            variant,
            group="rnn",
            status="ok",
            run_id=f"run-{variant.label}",
            metrics={"val_macro_f1": 0.5, "test_macro_f1": 0.4},
        )
        for variant in ablation.GROUPS["rnn"]
    ]
    payload = ablation.payload(
        "rnn", records, smoke=True, epochs=1, device="cpu", window_data_dirs={}
    )
    path = ablation.write_ablation(tmp_dir, payload)
    assert path.name == "ablation_rnn.json"
    loaded = ablation.load_ablation(path)
    assert loaded["group"] == "rnn"
    assert loaded["smoke"] is True
    assert loaded["reference"]["label"] == "cnn"
    assert len(loaded["runs"]) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _row(run_id: str, test_f1: str) -> dict[str, str]:
    return {
        "run_id": run_id,
        "model": "cnn",
        "window_s": "3.0",
        "aug": "basic",
        "pretrained": "none",
        "pool": "",
        "n_train": "10",
        "n_val": "5",
        "n_test": "5",
        "val_macro_f1": "0.500",
        "test_macro_f1": test_f1,
        "test_top1": "0.400",
        "test_top3": "0.700",
        "seed": "42",
        "device": "cpu",
        "git_sha": "abcdef1234567890",
        "notes": "smoke",
    }


ROWS = [_row("run-a", "0.400"), _row("run-b", "0.450"), _row("run-c", "")]


def test_select_rows_keeps_requested_order() -> None:
    selected = select_rows(ROWS, ["run-c", "run-a"])
    assert [row["run_id"] for row in selected] == ["run-c", "run-a"]


def test_select_rows_rejects_unknown_run() -> None:
    with pytest.raises(DataError, match="brak runów"):
        select_rows(ROWS, ["run-a", "run-x"])


def test_render_table_has_all_runs_and_delta() -> None:
    table = render_table(ROWS, "run-a")
    assert table.count("\n") == len(ROWS) + 1  # header + separator + rows
    assert "run-a" in table and "run-b" in table and "run-c" in table
    assert "+0.050" in table
    assert "| 0.400 |" in table
    assert "n/d" in table  # run-c has no test metric


def test_render_report_is_marked_generated_and_keeps_conclusions_for_human() -> None:
    text = render_report(ROWS, reference_id="run-a")
    assert "nie edytuj tabeli ręcznie" in text
    assert "## 5. Wnioski" in text
    assert "uzupełnia człowiek" in text
    for row in ROWS:
        assert row["run_id"] in text
    assert "Pary mylące" in text


def test_write_report_creates_file(tmp_dir: Path) -> None:
    path = write_report(tmp_dir / "docs" / "experiments.md", ROWS, reference_id="run-a")
    assert path.is_file()
    assert "Tabela główna" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI end-to-end on the tiny fixture (offline, smoke)
# --------------------------------------------------------------------------- #
def _argv(tmp_dir: Path, *rest: str) -> list[str]:
    return [
        "--env-file",
        str(tmp_dir / "nonexistent.env"),
        "--data-dir",
        str(tmp_dir / "data"),
        "--runs-dir",
        str(tmp_dir / "runs"),
        "--device",
        "cpu",
        *rest,
    ]


def _seed_trainable(tmp_dir: Path) -> Path:
    data_dir = tmp_dir / "data"
    make_catalog_with_audio(data_dir, per_species=4)
    assert main(_argv(tmp_dir, "prep", "cache")) == 0
    assert main(_argv(tmp_dir, "data", "split")) == 0
    return data_dir


def test_ablate_rnn_smoke_writes_runs_and_definition(tmp_dir: Path) -> None:
    _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "ablate", "--group", "rnn")) == 0
    payload = ablation.load_ablation(tmp_dir / "runs" / "ablation_rnn.json")
    assert [record["status"] for record in payload["runs"]] == ["ok", "ok", "ok"]
    for record in payload["runs"]:
        assert len(record["diff"]) <= 1
        assert record["run_id"]
    rows = load_index(tmp_dir / "runs")
    assert len(rows) == 3
    assert {row["model"] for row in rows} == {"cnn", "crnn"}


def test_report_cli_writes_generated_table(tmp_dir: Path) -> None:
    _seed_trainable(tmp_dir)
    assert main(_argv(tmp_dir, "--smoke", "ablate", "--group", "pool")) == 0
    payload = ablation.load_ablation(tmp_dir / "runs" / "ablation_pool.json")
    run_ids = [record["run_id"] for record in payload["runs"]]
    out = tmp_dir / "docs" / "experiments.md"
    assert main(_argv(tmp_dir, "report", "--runs", *run_ids, "--out", str(out))) == 0
    text = out.read_text(encoding="utf-8")
    assert "nie edytuj tabeli ręcznie" in text
    for run_id in run_ids:
        assert run_id in text
