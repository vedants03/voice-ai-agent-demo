# Flexi Term Pro — Bilingual Voice AI Agent (LiveKit)

A demo voice agent, **"Priya"**, that answers questions about the *Bharti AXA Life
Flexi Term Pro* insurance brochure over voice, in **English or Hindi** (auto-detected),
grounded in the document. All speech + language models run through **LiveKit Inference**
(only `LIVEKIT_*` keys needed — no Deepgram/OpenAI/ElevenLabs accounts). Demoed in the
**LiveKit Agents Playground** — no telephony.

## What it does
- 🎙️ Real-time voice conversation (STT → LLM → TTS) on the LiveKit free tier.
- 🌐 **Bilingual:** replies in whichever language the caller uses — English, Hindi, or Hinglish.
- 📄 **Grounded:** answers come from the brochure, including the premium/benefit **tables**.
- 🧑‍💼 **Scripted persona:** warm advisor "Priya" — greeting → guided Q&A → closing.

## Pipeline

| Stage | Model (via LiveKit Inference) | Notes |
|-------|-------------------------------|-------|
| STT   | `deepgram/nova-3` (`language="multi"`) | Hindi + English + Hinglish |
| LLM   | `google/gemini-2.5-flash`     | multilingual, large context, low latency |
| TTS   | `elevenlabs/eleven_flash_v2_5`| multilingual, fast; female voice "Jessica" |
| VAD   | Silero                        | local, tuned `min_silence_duration` |
| Turn  | `MultilingualModel`           | end-of-turn detection incl. Hindi |

## How the knowledge is handled — two approaches I explored

**Approach 1 — RAG (vector retrieval).** Chunk the PDF (table-aware), embed with a
local multilingual model, and at each turn retrieve the top-k chunks and inject them.
Code: [`src/ingest.py`](src/ingest.py) (build index) + [`src/rag.py`](src/rag.py) (retrieve).

This **underperformed for this document**, for concrete reasons:
- The whole brochure is only ~10k tokens (122 chunks), but top-k=4 retrieval injected
  only ~4 chunks per turn — throwing away ~95% of a document that easily fits in context.
- The complex multi-header premium tables extracted as garbled text, so even when a
  table chunk *was* retrieved it wasn't legible → "can't answer tabular data".
- In a live call, **STT noise** in the spoken query poisoned the retrieval lookup, so
  the agent often returned "I don't have that information" (worse live than in offline tests).

**Approach 2 — full-context injection (current).** Since the brochure is small, inject the
**entire curated knowledge base** ([`data/knowledge.md`](data/knowledge.md), with tables
rebuilt as clean markdown) into the system prompt every turn. The model always sees every
fact, retrieval can't "miss", and STT noise is survivable because the model reasons with
the whole document in view. Code: [`src/agent.py`](src/agent.py).

**Takeaway:** RAG earns its complexity when the corpus is too big for the context window.
At ~10k tokens, retrieval was a bottleneck that could only lose information. Both code paths
are kept here to show the comparison.

## Latency tuning
The "user stops speaking → agent replies" gap is dominated by **endpointing**, not TTS.
Tuned in [`src/agent.py`](src/agent.py):
- `preemptive_generation=True` — LLM starts generating during the endpointing wait.
- `min_endpointing_delay=0.3` (from 0.5s) and Silero `min_silence_duration=0.4` (from 0.55s).

These trade snappiness against the risk of interrupting a caller who pauses — raise them
slightly if Priya cuts you off; lower them if she still feels slow.

## Project structure
```
voice-ai-agent-demo/
  data/
    flexi-term-pro.pdf      # source brochure
    knowledge.md            # curated knowledge base injected into the prompt (Approach 2)
  src/
    agent.py                # the voice agent (full-context, persona, latency tuning)
    eval_qa.py              # headless QA eval — accuracy incl. tables, EN/HI/Hinglish
    voice_test.py           # synthesize female-voice samples to pick Priya's voice
    ingest.py               # Approach 1: build the RAG vector index
    rag.py                  # Approach 1: load index + cosine top-k retrieval
  requirements.txt
  .env.example              # copy to .env.local and add your LiveKit keys
```

## Setup
Requires **Python 3.12** on PATH (`python --version` → 3.12.x).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use: source .venv/bin/activate on macOS/Linux)
python -m pip install -r requirements.txt
copy .env.example .env.local      # then add your LiveKit Cloud keys
python src\agent.py download-files   # pre-download Silero + turn-detector models (once)
```

Get free LiveKit Cloud keys at https://cloud.livekit.io → project → Settings → Keys.

## Run the demo
```bash
python src\agent.py dev
```
Wait for `registered worker`, then open **https://agents-playground.livekit.io**, connect
to the same LiveKit project, click the mic, and talk. Priya greets you in English; ask in
English or Hindi.

> Stop with **Ctrl+C** (graceful) so worker subprocesses exit cleanly.

### Try asking
- "What is the minimum sum assured?"  /  "What's the maximum maturity age?"
- "What's the yearly premium for a 30 year old male for 1 crore cover?"  *(table lookup)*
- "इस प्लान में मृत्यु पर क्या लाभ मिलता है?"  *(death benefit, Hindi)*
- "Premium payment options kya kya hain?"  *(Hinglish)*

## Verification (no mic needed)
- **Accuracy / bilingual:** `python src\eval_qa.py` — runs EN/Hindi/Hinglish + table-lookup
  questions through the prompt + Inference LLM and prints grounded answers.
- **Voice:** `python src\voice_test.py` — synthesizes English+Hindi samples for the
  candidate female voices so you can pick Priya's voice.

## Notes & limitations
- **Demo scope:** single brochure, no telephony, runs locally + the LiveKit Playground.
- **Voice accent:** LiveKit Inference only serves ElevenLabs *default* voices, so Priya uses
  an American/British female voice (Hindi is spoken with that accent). Indian-accent community
  voices would need a direct ElevenLabs account + plugin.
- **Free tier:** ~$2.50 Inference credits + 1,000 agent-min/month — ample for a demo.
- Not financial advice — for quotes/purchase the agent directs callers to an advisor.
