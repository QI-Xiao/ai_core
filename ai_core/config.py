"""Env-driven defaults. Loaded once from process env / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from CWD if present. Each project decides where its .env lives;
# uvicorn is typically launched from the project root.
load_dotenv(Path.cwd() / ".env")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_DEFAULT_MODEL: str = "claude-sonnet-4-6"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL: str = "gpt-4o"

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL: str = "llama3.2"

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8002")
VLLM_DEFAULT_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

AI_DEFAULT_MODEL: str = os.getenv("AI_DEFAULT_MODEL", "")

_raw_origins = os.getenv("AI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]
