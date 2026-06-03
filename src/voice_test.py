"""Verify the chosen TTS voice works via LiveKit Inference and sounds right.
Synthesizes a short English+Hindi line and writes voice_sample.wav so you can
listen. Usage: ``python src\\voice_test.py``
"""

from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import inference

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")

TEXT = (
    "Hello, I'm Priya from Bharti AXA Life. I can help you with the Flexi Term Pro plan. "
    "Namaste, main aapki kaise madad kar sakti hoon?"
)

# Female voices confirmed available via LiveKit Inference (ElevenLabs default set).
CANDIDATES = {
    "jessica": "cgSgspJ2msm6clMCkdW9",  # American female
    "alice": "Xb7hH8MSUJpSbSDYk0k2",  # British female
}


async def synth(name: str, voice_id: str) -> None:
    out = ROOT / f"voice_sample_{name}.wav"
    tts = inference.TTS(model="elevenlabs/eleven_flash_v2_5", voice=voice_id)
    frames = []
    try:
        async with tts.synthesize(TEXT) as stream:
            async for ev in stream:
                frames.append(ev.frame)
    except Exception as e:
        print(f"  {name} ({voice_id}): FAILED — {e}")
        return
    if not frames:
        print(f"  {name} ({voice_id}): no audio produced")
        return
    sr, ch = frames[0].sample_rate, frames[0].num_channels
    pcm = b"".join(bytes(f.data) for f in frames)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    secs = len(pcm) / (sr * ch * 2)
    print(f"  {name} ({voice_id}): OK ~{secs:.1f}s -> {out.name}")


async def main() -> None:
    print("Generating female-voice samples (English + Hindi):")
    for name, vid in CANDIDATES.items():
        await synth(name, vid)
    print("\nPlay the .wav files and tell me which voice you prefer for Priya.")


if __name__ == "__main__":
    asyncio.run(main())
