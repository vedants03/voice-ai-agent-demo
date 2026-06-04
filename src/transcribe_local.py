"""One-off local transcription with faster-whisper (no API / rate limits).
Handles English + Hindi. Usage: ``python src\\transcribe_local.py "path\\to\\call.mp3" [model]``
(model default "small"; use "medium" for better Hindi).
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from faster_whisper import WhisperModel


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python src\\transcribe_local.py "audio.mp3" [model]')
        return
    path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"

    print(f"loading whisper '{model_size}'...", file=sys.stderr)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, beam_size=5, vad_filter=True)
    print(f"detected language: {info.language} (p={info.language_probability:.2f})", file=sys.stderr)

    for seg in segments:
        print(f"[{seg.start:6.1f}-{seg.end:6.1f}] {seg.text.strip()}")


if __name__ == "__main__":
    main()
