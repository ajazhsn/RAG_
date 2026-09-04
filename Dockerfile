# Serving image for the RAG Eval API.
# Built and validated in CI (see .github/workflows/ci.yml) — proves the
# container builds and starts cleanly on every push, without needing to
# spend API quota inside CI itself.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-embed.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-embed.txt fastapi uvicorn[standard]

COPY src/ src/
COPY config/ config/
RUN pip install --no-cache-dir -e .

# data/ is intentionally NOT copied — mount it as a volume at runtime, since
# embeddings/corpus are large and shouldn't bloat the image. This matches how
# a real deployment would separate code (image) from data (volume/object storage).

EXPOSE 8000

CMD ["uvicorn", "rag_eval.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
