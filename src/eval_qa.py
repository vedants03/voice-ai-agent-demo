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

QUESTIONS = [
    "What is the minimum sum assured?",  # table
    "What's the yearly premium for a 30 year old male for 1 crore cover, single life?",  # sample premium table
    "What are the death benefit payout options?",
    "Which riders can I add to this plan?",  # rider table
    "What is the maximum age at maturity?",  # table
    "इस प्लान में एंट्री के लिए न्यूनतम और अधिकतम उम्र क्या है?",  # entry age (Hindi, table)
    "Ek 35 saal ke aadmi ke liye 1 crore ka premium kitna hai?",  # Hinglish, table
    "Does this plan cover COVID hospitalization?",  # edge: hospi cash rider exists, COVID not named
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
