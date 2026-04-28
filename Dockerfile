# Base image for ai_core consumers.
#
# Each project Dockerfile should:
#   FROM ai-core:latest
#   COPY app /srv/app
#   COPY rag_knowledge /srv/rag_knowledge   # optional
#   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
#
# Build with: docker build -t ai-core:latest /Users/xiao.qi/Projects/ai_core
FROM python:3.13-slim

WORKDIR /srv

COPY pyproject.toml /tmp/ai_core/
COPY ai_core /tmp/ai_core/ai_core
RUN pip install --no-cache-dir /tmp/ai_core && rm -rf /tmp/ai_core

EXPOSE 8001
# Default CMD assumes the consuming image places its app at /srv/app/main.py
# exposing a FastAPI app named `app`. Override in project Dockerfile if needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
