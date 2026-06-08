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

from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")

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

# Style (this is a live voice call)
- Keep turns **short and natural — usually 1 to 2 sentences.** Don't monologue or info-dump.
- Be **consultative and polite, never pushy** — respect their time; if they decline, be gracious.
- Don't read markdown, lists, or symbols aloud. Say dates and times naturally ("Friday at 3 PM").
- Speak any numbers naturally; read phone numbers digit by digit in the language you're speaking.
- **Do not invent** pricing, guarantees, specific case-study numbers, or services beyond the list
  above — offer to cover specifics in the meeting.
- Always keep steering, gently, toward booking the 15-minute meeting.
"""

OPENING = (
    f"You are {AGENT_NAME} from {COMPANY} making an outbound call. Open in ENGLISH: warmly "
    "confirm you're speaking with the right person, introduce yourself and the company in one "
    "line (an end-to-end marketing and sales partner helping with cleaner contact data, lead "
    "generation, appointment setting, SEO and paid ads), and ask your first discovery question "
    "about whether their team faces challenges reaching the right decision-makers or with "
    "lead-data quality. Keep it to two short sentences."
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
        stt=inference.STT(model="deepgram/nova-3", language="multi"),  # Hindi+English
        # Measured: TTFT here is network/gateway-bound, not model-bound — all models
        # land ~1.1-1.5s from a local (India) machine. flash was fastest + best quality,
        # so flash-lite gave no benefit. The real latency lever is co-locating the agent
        # with the gateway (deploy to LiveKit Cloud) rather than running dev locally.
        llm=inference.LLM(model="google/gemini-2.5-flash"),
        tts=inference.TTS(model="elevenlabs/eleven_flash_v2_5", voice=TTS_VOICE),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,  # generate during the endpointing wait
        min_endpointing_delay=0.2,  # shorter post-speech wait (default 0.5s)
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

    await session.generate_reply(instructions=OPENING)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
