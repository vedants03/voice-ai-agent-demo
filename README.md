# Bilingual Outbound Sales Voice Agent (LiveKit)

A real-time **outbound B2B sales / appointment-setting** voice agent. It opens a cold call,
probes the prospect's pain points, handles questions and objections, and books a short
discovery meeting. **Bilingual** — English-first, switching to Hindi/Hinglish if the prospect
does. Demoed in the **LiveKit Agents Playground** (no telephony).

## Stack

| Stage | Model / Service |
|-------|-----------------|
| **STT** | Deepgram Nova-3 (multilingual) via LiveKit Inference |
| **LLM** | Google Gemini 2.5 Flash on Vertex AI (`asia-south1`) |
| **TTS** | ElevenLabs Flash v2.5 (multilingual) via LiveKit Inference |
| **Turn-taking** | Silero VAD |
| **Orchestration** | LiveKit Agents |

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

## Caveats

- **LLM** runs on Vertex AI and requires a Google Cloud service account with Vertex AI access.
- **STT & TTS** run via LiveKit Inference (US-hosted), which adds network latency from other regions.
- **Voice** is an ElevenLabs default voice (single accent); it speaks Hindi with that voice's accent.
- **Demo scope:** placeholder identity, verbal meeting booking (no live calendar), free-tier credits.

## Optional: deploy to LiveKit Cloud

A `Dockerfile` is included to run the agent as a managed LiveKit Cloud agent (`lk agent create`).
Secrets are injected by the platform at runtime.

## Project structure

```
src/agent.py        # the voice agent (STT/LLM/TTS pipeline, prompt, turn-taking)
Dockerfile          # optional LiveKit Cloud deployment
requirements.txt
.env.example        # copy to .env.local and add credentials
```

> The repo also contains an earlier document-retrieval (RAG) experiment — `src/ingest.py`,
> `src/rag.py`, and `data/` — kept for reference; it is not used by this agent.
