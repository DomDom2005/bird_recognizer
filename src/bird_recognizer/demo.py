"""Gradio demo: record or upload a recording, get top-3 species and the spectrogram (FR-13).

Runs on CPU, without network. The model is loaded from a training run exactly like
``eval`` does (run's own config and checkpoint). ``--smoke`` never starts the server:
it scores a synthetic tone and checks that the whole path returns a top-3, which is
enough to prove the demo wiring works without pretending the tone is a bird.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .data.audio import load_audio
from .data.spectrogram import mel_spectrogram
from .data.windows import iter_window_signals, plan_windows, valid_frames
from .errors import DataError
from .eval.evaluate import RunBundle, model_probs, species_names

LOGGER = logging.getLogger(__name__)

DISCLAIMER = (
    "**To wynik eksperymentalny.** Model rozpoznaje 12 gatunków polskich ptaków "
    "śpiewających, uczony na nagraniach z xeno-canto (jeden kraj, mikrofony kierunkowe). "
    "Nie służy do identyfikacji gatunków chronionych ani do decyzji terenowych."
)


def ranking(bundle: RunBundle, probabilities: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
    names = species_names(bundle.config)
    scored = [
        (names[index], float(probabilities[index])) for index in range(len(probabilities))
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top]


def predict_file(
    bundle: RunBundle, path: Path
) -> tuple[list[tuple[str, float]], list[float], np.ndarray]:
    """Score one audio file; returns top-3, kept-window starts and the raw signal."""
    signal = load_audio(Path(path), bundle.config.audio)
    plan = plan_windows(signal, bundle.config.audio, bundle.config.gating)
    frames: list[np.ndarray] = []
    lengths: list[int] = []
    starts: list[float] = []
    for index, window in iter_window_signals(signal, plan, only_kept=True):
        frames.append(
            mel_spectrogram(window, bundle.config.audio.sample_rate, bundle.config.spectrogram)
        )
        lengths.append(valid_frames(plan.lengths[index], bundle.config.spectrogram.hop_length))
        starts.append(round(plan.starts[index] / bundle.config.audio.sample_rate, 3))
    if not frames:
        raise DataError("nagranie nie ma okien, które przeszły bramkowanie ciszy")
    probabilities = model_probs(
        bundle.model, np.stack(frames).astype(np.float32), lengths, bundle.device
    )
    return ranking(bundle, probabilities.mean(axis=0)), starts, signal


def render_spectrogram(bundle: RunBundle, signal: np.ndarray, starts: list[float]):
    """Matplotlib figure of the log-mel map with kept windows marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = mel_spectrogram(signal, bundle.config.audio.sample_rate, bundle.config.spectrogram)
    frames_per_second = (
        bundle.config.audio.sample_rate / bundle.config.spectrogram.hop_length
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    for start in starts:
        axis.axvline(
            start * frames_per_second,
            color="cyan",
            linewidth=0.8,
            alpha=0.9,
        )
    axis.set_xlabel("czas [s]")
    axis.set_ylabel("pasmo mel")
    axis.set_title("log-mel; pionowe linie = zachowane okna")
    figure.tight_layout()
    return figure


def model_card(bundle: RunBundle) -> str:
    options = ", ".join(f"{key}={value}" for key, value in bundle.options.items()) or "—"
    metrics_file = bundle.directory / "metrics.json"
    parameters = ""
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics.get("parameters"):
            parameters = f", {int(metrics['parameters'])} parametrów"
    return (
        f"**Model:** `{bundle.model_name}` ({options}){parameters} · run `{bundle.run_id}` "
        f"· seed {bundle.config.seed} · okno {bundle.config.audio.window_s:g} s "
        f"· urządzenie {bundle.device}\n\n{DISCLAIMER}"
    )


def predict_for_ui(bundle: RunBundle, path: str | None):
    """Gradio callback: never raises — returns a readable message instead."""
    if not path:
        return {}, None, "Wgraj plik audio (wav, mp3, ogg, flac)."
    try:
        top, starts, signal = predict_file(bundle, Path(path))
    except (DataError, OSError) as exc:
        return {}, None, f"**BŁĄD:** {exc}"
    figure = render_spectrogram(bundle, signal, starts)
    return {name: probability for name, probability in top}, figure, model_card(bundle)


def build_app(bundle: RunBundle):
    """Build (but do not launch) the Gradio Blocks app."""
    # Gradio pings analytics endpoints on launch; the demo must stay offline (FR-13).
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    with gr.Blocks(title="bird_recognizer") as app:
        gr.Markdown(
            "# bird_recognizer — demo\n"
            "Nagraj ptaka mikrofonem albo wgraj plik; model zwraca top-3 gatunki "
            "i spektrogram z oknami."
        )
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Nagraj lub wybierz nagranie (wav / mp3 / ogg / flac)",
        )
        with gr.Row():
            label = gr.Label(num_top_classes=3, label="Top-3 gatunki")
            plot = gr.Plot(label="Spektrogram")
        info = gr.Markdown(model_card(bundle))
        audio.change(
            fn=lambda path: predict_for_ui(bundle, path),
            inputs=audio,
            outputs=[label, plot, info],
        )
    return app


def launch(bundle: RunBundle, *, port: int = 7860) -> None:
    app = build_app(bundle)
    app.launch(server_port=port, show_error=True)


def _synthetic_wav(path: Path) -> Path:
    """Deterministic 6 s tone; only for ``--smoke``, never presented as a bird."""
    import soundfile as sf

    sample_rate = 32000
    timeline = np.arange(int(6.0 * sample_rate), dtype=np.float64) / sample_rate
    signal = (0.3 * np.sin(2.0 * np.pi * 1000.0 * timeline)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate)
    return path


def smoke_check(bundle: RunBundle) -> int:
    """Score a synthetic tone and print its top-3; proves the demo path works."""
    target = Path(bundle.config.paths.data_dir) / ".demo_smoke.wav"
    try:
        path = _synthetic_wav(target)
        top, starts, _signal = predict_file(bundle, path)
    finally:
        target.unlink(missing_ok=True)
    print(f"Demo smoke (run {bundle.run_id}, device={bundle.device}):")
    print(f"  plik syntetyczny: 6 s ton 1000 Hz, okien: {len(starts)}")
    for name, probability in top:
        print(f"  {name}: {probability:.3f}")
    print("  UWAGA: to nie jest wynik badania — test okablowania demo.")
    return 0
