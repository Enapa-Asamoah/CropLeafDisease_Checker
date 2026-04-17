from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from typing import Optional, Tuple


def _normalize_lang_code(lang_code: Optional[str]) -> str:
    code = (lang_code or "").strip().lower()
    if code.startswith("en"):
        return "en"
    if code in {"ak", "tw", "twi"}:
        return "ak"
    return "en"


@lru_cache(maxsize=1)
def _load_stt_model():
    from faster_whisper import WhisperModel

    # CPU int8 keeps this usable on most laptops.
    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio_bytes(audio_bytes: bytes) -> Tuple[str, str]:
    """Return (transcript, detected_language_code)."""
    if not audio_bytes:
        return "", ""

    model = _load_stt_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, info = model.transcribe(tmp.name, beam_size=1, vad_filter=True)

    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    detected = getattr(info, "language", "") or ""
    return transcript, detected


@lru_cache(maxsize=2)
def _load_tts_stack(lang_code: str):
    import torch
    from transformers import AutoTokenizer, VitsModel

    normalized = _normalize_lang_code(lang_code)
    model_name = "facebook/mms-tts-aka" if normalized == "ak" else "facebook/mms-tts-eng"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def synthesize_speech_bytes(text: str, lang_code: Optional[str] = "en") -> bytes:
    """Generate WAV bytes from text for English or Twi (Akan)."""
    clean_text = (text or "").strip()
    if not clean_text:
        return b""

    import numpy as np
    import soundfile as sf
    import torch

    normalized = _normalize_lang_code(lang_code)
    tokenizer, model = _load_tts_stack(normalized)

    inputs = tokenizer(clean_text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().cpu().numpy()

    # Keep output stable and avoid clipping.
    waveform = np.clip(waveform, -1.0, 1.0)

    sample_rate = int(getattr(model.config, "sampling_rate", 16000))
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()


def speech_dependencies_ready() -> Tuple[bool, str]:
    try:
        import faster_whisper  # noqa: F401
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, ""
