# Bilingual Outbound Sales Voice Agent (LiveKit)

A real-time **outbound B2B sales / appointment-setting** voice agent. It opens a cold call,
probes the prospect's pain points, handles questions and objections, and books a short
discovery meeting. **Bilingual** — English-first, switching to Hindi/Hinglish if the prospect
does. Demoed in the **LiveKit Agents Playground** (no telephony).

## Stack

| Stage | Model / Service | Notes |
|-------|-----------------|-------|
| **STT** | Deepgram Nova-3 (multilingual) via LiveKit Inference | Hindi + English + Hinglish |
| **LLM** | Google Gemini 2.5 Flash on Vertex AI (`asia-south1`) | "thinking" disabled for low latency |
| **TTS** | ElevenLabs Flash v2.5 (multilingual) via LiveKit Inference | streaming |
| **Turn-taking** | Silero VAD + VAD-based endpointing | low latency; ignores short filler words |
| **Orchestration** | LiveKit Agents | per-turn latency metrics |

## Latency

Per-turn latency is instrumented (end-of-utterance, STT, LLM TTFT, TTS TTFB) and logged. Key
optimizations, in order of impact:

- **Disabled the LLM's "thinking" pass** — cut first-token time ~3× (≈2,000ms → ≈400ms).
- **Region-local LLM** — Gemini on Vertex AI in-region instead of a distant gateway.
- **VAD-based turn detection** — end-of-turn ≈ 0.5s vs ~1.25s with a semantic detector.
- **Preemptive generation** — the LLM starts generating during the endpointing wait.

## Conversation behavior

- English-first; mirrors the prospect into Hindi/Hinglish when they switch.
- Short, natural turns (one to two sentences) — tuned against over-explaining.
- Short filler words / backchannels ("uh-huh", "okay", "haan") don't interrupt the agent.

## Setup

Requires **Python 3.12** on PATH.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env.local      # then fill in credentials (below)
python src\agent.py download-files   # pre-download VAD model files (once)
```

Credentials in `.env.local`:

- **LiveKit Cloud** (free tier — covers STT & TTS via Inference):
  `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — from https://cloud.livekit.io → project → Settings → Keys.
- **Google Cloud** (Vertex AI — for the Gemini LLM):
  `GOOGLE_APPLICATION_CREDENTIALS` (path to a service-account key with Vertex AI access),
  `GCP_PROJECT`, and `GCP_LOCATION` (e.g. `asia-south1`).

## Run

```bash
python src\agent.py dev
```

Wait for `registered worker`, then open **https://agents-playground.livekit.io**, connect to
your project, and talk to the agent. Stop with **Ctrl+C** (graceful) so worker subprocesses
exit cleanly.

## Caveats / trade-offs

- **LLM** runs on Vertex AI (needs a GCP service account); "thinking" is disabled for speed,
  trading deep multi-step reasoning — not needed for this conversational task.
- **STT & TTS** run via LiveKit Inference (US-hosted), adding ~0.3–0.5s each; region-local
  services would reduce latency further.
- **Voice** is an ElevenLabs default voice (single accent); LiveKit Inference doesn't expose
  custom/community voices. It speaks Hindi but with that voice's accent.
- **Turn-taking** is VAD-based for low latency, so it may occasionally clip a caller who pauses
  mid-sentence (tunable via VAD silence + interruption settings).
- **Demo scope:** placeholder identity, verbal meeting booking (no live calendar), free-tier credits.

## Optional: deploy to LiveKit Cloud

A `Dockerfile` is included to run the agent as a managed LiveKit Cloud agent (`lk agent create`),
co-located with the gateway. Secrets are injected by the platform at runtime.

## Project structure

```
src/agent.py        # the voice agent (STT/LLM/TTS pipeline, prompt, turn-taking)
Dockerfile          # optional LiveKit Cloud deployment
requirements.txt
.env.example        # copy to .env.local and add credentials
```

> The repo also contains an earlier document-retrieval (RAG) experiment — `src/ingest.py`,
> `src/rag.py`, and `data/` — kept for reference; it is not used by this agent.
