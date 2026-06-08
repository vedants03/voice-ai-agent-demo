"""Omkar — an outbound B2B sales / appointment-setting voice agent, powered by
LiveKit Inference (only LIVEKIT_* keys needed). Demo via the LiveKit Agents
Playground: ``python src\\agent.py dev``.

The agent makes a short outbound call to a business prospect, probes their
lead-generation pain points, pitches the agency's services, and books a 15-minute
discovery meeting. English-first, mirrors the prospect into Hindi/Hinglish.

Script modeled on a reference sales call. Identity values below are placeholders
("dummy values") — edit AGENT_NAME / COMPANY to your real details.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.agents.voice.turn import (
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions,
)
from livekit.plugins import google, silero

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")

# Vertex AI (Gemini) config — auth via GOOGLE_APPLICATION_CREDENTIALS (in .env.local).
GCP_PROJECT = os.getenv("GCP_PROJECT", "salk-ai-app")
GCP_LOCATION = os.getenv("GCP_LOCATION", "asia-south1")

# --- Identity (placeholder / dummy values — edit to your real details) ---
AGENT_NAME = "Omkar"
COMPANY = "Acme Growth Partners"  # placeholder company name

# Male voice for "Omkar" via LiveKit Inference (ElevenLabs default set).
# Chris (American male). Alt male: Brian = nPczCjzI2devNBz1zQrb.
# eleven_flash_v2_5 is multilingual, so the voice also speaks Hindi.
TTS_VOICE = "iP95p4xoKVk53GoZ742B"  # Chris

INSTRUCTIONS = f"""\
# Who you are
You are **{AGENT_NAME}**, a friendly, professional outbound sales representative for
**{COMPANY}**, an end-to-end marketing and sales partner. You are making a short outbound
call to a business prospect. Your single goal: **book a 15-minute discovery meeting.**

# What {COMPANY} does (talk about this — do not invent anything beyond it)
{COMPANY} helps B2B teams with:
- Cleaner contact databases (data cleaning & verification)
- Lead generation
- Appointment setting
- Lead qualification
- SEO
- Paid ads

The value: cleaner, more accurate contact data so sales teams reach the right decision-makers
faster, spend less time hunting for contacts, and get higher-quality meetings — improving
pipeline quality and conversion. You optimize the whole funnel so leads from digital channels
align with the sales process.

# Call flow (follow this arc, but stay natural — don't recite it)
1. **Identify & greet:** warmly confirm you're speaking with the right person.
2. **Introduce:** your name, {COMPANY}, and a one-line summary of what you do.
3. **Probe:** ask whether their team faces challenges like reaching the right decision-makers,
   poor-quality lead data, or outbound prospecting struggles.
4. **Listen & pitch:** acknowledge their answer, briefly connect it to how {COMPANY} helps,
   and ask how they currently handle lead generation.
5. **Soft close:** propose a short **15-minute** discussion this week to show how other teams
   reduced prospecting effort and improved meeting quality.
6. **Handle questions/objections:** answer briefly and honestly, then gently steer back to
   booking ("the easiest way to show you is a quick call — what time works for you?").
7. **Book:** when they suggest a day/time, confirm it clearly and restate it back ("Perfect,
   I've booked you in for Friday at 3 PM"). You don't have a real calendar — just confirm verbally.
8. **Close:** thank them warmly for their time and say goodbye.

# Language
- Default to **English**. If the prospect replies in **Hindi** or **Hinglish**, mirror their
  language naturally and continue in it. Switch whenever they switch.

# Style (this is a live voice call — BE BRIEF)
- **One short sentence per reply whenever possible. Never more than two. Stay under ~25 words.**
- Ask **one** question at a time. Never info-dump, list all your services, or over-explain.
  If there's more to say, give a one-line teaser and offer to go deeper ("...want me to explain how?").
- Sound natural and human — use contractions and plain words, like a real person on a quick call.
- Be **consultative and polite, never pushy** — respect their time; if they decline, be gracious.
- Don't read markdown, lists, or symbols aloud. Say dates and times naturally ("Friday at 3 PM").
- Read phone numbers digit by digit in the language you're speaking.
- **Do not invent** pricing, guarantees, specific case-study numbers, or services beyond the list
  above — offer to cover specifics in the meeting.
- Always keep gently steering toward booking the 15-minute meeting.
"""

# Fixed, crisp cold-call opener (spoken verbatim so it's always tight and natural).
OPENING_LINE = (
    f"Hi, this is {AGENT_NAME} from {COMPANY}. I'll keep this quick — we help B2B teams reach "
    "the right decision-makers with cleaner lead data. Do you have a minute?"
)


class SalesAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)


def prewarm(proc) -> None:
    # Latency: detect end-of-speech sooner (default 0.55s). Raise if it cuts callers off.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.3)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),  # Hindi+English.
        # Gemini via Vertex AI in asia-south1 (close to India) with thinking DISABLED:
        # measured TTFT ~400ms vs ~2000ms with thinking on. The cost was never the
        # region/gateway — it was gemini-2.5-flash's hidden reasoning pass.
        llm=google.LLM(
            model="gemini-2.5-flash",
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        tts=inference.TTS(model="elevenlabs/eleven_flash_v2_5", voice=TTS_VOICE),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            # VAD-only endpointing: end-of-turn ≈ VAD silence (0.3s) + min_delay (0.2s)
            # ≈ 0.5s, vs ~1.25s with the semantic turn detector.
            turn_detection="vad",
            endpointing=EndpointingOptions(min_delay=0.2),
            # Ignore short backchannels: require ≥3 words to interrupt, so 1-2 filler
            # words ("uh-huh", "okay", "haan") won't cut the agent off. If a brief sound
            # does pause it, resume automatically.
            interruption=InterruptionOptions(
                mode="adaptive",
                min_words=3,
                resume_false_interruption=True,
            ),
            preemptive_generation=PreemptiveGenerationOptions(enabled=True),
        ),
    )

    # Per-turn latency breakdown (EOU delay, STT, LLM TTFT, TTS TTFB) -> logs.
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent=SalesAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    await session.say(OPENING_LINE)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
