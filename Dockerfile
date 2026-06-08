# LiveKit Cloud agent image (Python 3.12). Builds the worker that connects to
# the project in "start" (production) mode. LIVEKIT_* secrets are injected by
# LiveKit Cloud at runtime — they are NOT baked into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build tools for any source wheels (PyAV/onnxruntime ship prebuilt wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY . .

# Pre-download Silero VAD + multilingual turn-detector model files at build time
# so cold starts are fast.
RUN python src/agent.py download-files

# Run as non-root.
RUN adduser --disabled-password --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Production mode: connects to LiveKit Cloud and waits for jobs.
CMD ["python", "src/agent.py", "start"]
