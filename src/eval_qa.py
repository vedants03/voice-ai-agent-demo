"""Headless QA eval: run questions through the full-context agent prompt + the
LiveKit Inference LLM and print answers. Validates accuracy (incl. tables) and
bilingual answering without a mic/speaker. Usage: ``python src\\eval_qa.py``
(or pass your own questions as args).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import inference
from livekit.agents.llm import ChatContext

from agent import INSTRUCTIONS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

# Simulated prospect turns to sanity-check the outbound sales agent's responses,
# objection handling, redirect-to-booking, and bilingual mirroring (single-turn).
QUESTIONS = [
    "Sorry, who is this and what does your company do?",  # intro
    "We already get our leads from LinkedIn, we're good.",  # objection / current state
    "I'm pretty busy right now.",  # brush-off objection
    "How exactly would you help us improve our lead generation?",  # discovery question
    "Okay sure, let's do Friday at 3 PM.",  # booking -> should confirm verbally
    "Aap log kya karte ho exactly?",  # Hinglish: what do you do
    "हमें इसमें कोई दिलचस्पी नहीं है।",  # Hindi: not interested -> should be gracious
]


async def answer(llm: inference.LLM, question: str) -> str:
    ctx = ChatContext.empty()
    ctx.add_message(role="system", content=INSTRUCTIONS)
    ctx.add_message(role="user", content=question)
    out: list[str] = []
    async with llm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            delta = getattr(chunk, "delta", None)
            if delta and delta.content:
                out.append(delta.content)
    return "".join(out).strip()


async def main() -> None:
    questions = sys.argv[1:] or QUESTIONS
    llm = inference.LLM(model="google/gemini-2.5-flash")
    for q in questions:
        ans = await answer(llm, q)
        print(f"\nQ: {q}\nA: {ans}")


if __name__ == "__main__":
    asyncio.run(main())
