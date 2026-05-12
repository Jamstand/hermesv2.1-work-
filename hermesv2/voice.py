"""Voice-to-text using faster-whisper (local, CPU-friendly).

Why file-based, not live-mic: WSL audio passthrough is fragile (PulseAudio
proxy setup, Windows mic permissions, sample-rate mismatches). Taking an
audio file as input sidesteps all of that — record with Windows Voice
Recorder, drop the .wav/.m4a/.mp3 in, transcribe.

Install:
    pip install 'hermesv2[voice]'      # pulls faster-whisper
    # Models download lazily on first run into ~/.cache/huggingface/hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_DEFAULT_MODEL = "small"  # ~500 MB, decent quality on CPU
_MODEL_CACHE: dict[str, Any] = {}


def _get_model(name: str):
    """Lazy-load the Whisper model; cache it across calls in the same process."""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install 'hermesv2[voice]'"
        ) from e
    model = WhisperModel(name, device="cpu", compute_type="int8")
    _MODEL_CACHE[name] = model
    return model


def transcribe(
    audio_path: str | Path,
    model_name: str = _DEFAULT_MODEL,
    language: str | None = None,
) -> str:
    """Transcribe an audio file to plain text.

    Supports any format faster-whisper / ffmpeg can read (wav, mp3, m4a,
    ogg, flac, webm, etc.).
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model = _get_model(model_name)
    segments, _info = model.transcribe(str(path), language=language)
    return " ".join(seg.text.strip() for seg in segments).strip()
