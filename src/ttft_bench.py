"""Benchmark LLM time-to-first-token (TTFT) across Inference models on the sales
prompt, to pick the lowest-latency LLM. Usage: ``python src\\ttft_bench.py``
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import inference
from livekit.agents.llm import ChatContext

from agent import INSTRUCTIONS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

MODELS = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
    "openai/gpt-oss-120b",        # typically Cerebras/Groq-hosted, very fast
    "xai/grok-4-1-fast-non-reasoning",
]
USER_TURN = "We already get our leads from LinkedIn, we're good."
ITERS = 3


async def ttft_once(model: str) -> float:
    llm = inference.LLM(model=model)
    ctx = ChatContext.empty()
    ctx.add_message(role="system", content=INSTRUCTIONS)
    ctx.add_message(role="user", content=USER_TURN)
    t0 = time.perf_counter()
    async with llm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            delta = getattr(chunk, "delta", None)
            if delta and delta.content:
                return time.perf_counter() - t0
    return float("nan")


async def main() -> None:
    print(f"TTFT over {ITERS} iters (first token), sales prompt ~{len(INSTRUCTIONS)//4} tokens\n")
    for model in MODELS:
        times: list[float] = []
        for _ in range(ITERS):
            try:
                times.append(await ttft_once(model))
            except Exception as e:
                print(f"{model:42s} FAILED: {str(e)[:80]}")
                times = []
                break
        if times:
            best, avg = min(times) * 1000, sum(times) / len(times) * 1000
            print(f"{model:42s} best {best:6.0f}ms   avg {avg:6.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
