"""create_app() — FastAPI app factory parameterised by project-specific bits.

Typical usage from a project's main.py:

    from ai_core import create_app
    from app.tools.registry import HFR_TOOLS
    from app.prompts.system import HFR_SYSTEM_PROMPT

    app = create_app(
        tools=HFR_TOOLS,
        system_prompt=HFR_SYSTEM_PROMPT,
        knowledge_dir="rag_knowledge",
        plot_dir="plots",
    )
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai_core.api import chat as chat_api
from ai_core.api import health as health_api
from ai_core.api import models as models_api
from ai_core.config import CORS_ORIGINS
from ai_core.providers.router import DEFAULT_MODEL_CATALOG
from ai_core.tools.base import BaseTool


def create_app(
    *,
    tools: list[BaseTool],
    system_prompt: str | None = None,
    model_catalog: list[dict] | None = None,
    cors_origins: list[str] | None = None,
    plot_dir: str | Path | None = None,
    title: str = "AI Core",
    version: str = "0.1.0",
) -> FastAPI:
    """Build a FastAPI app exposing /ai/health, /ai/models, /ai/chat, /ai/chat/stream.

    Args:
        tools: Project-specific BaseTool instances. Pass [] for chat-only.
        system_prompt: System prompt sent to the model on every turn.
        model_catalog: Override the default model catalog (e.g. to remove providers).
        cors_origins: Override the default CORS origins (defaults to AI_CORS_ORIGINS env).
        plot_dir: If set, mount this directory at /plots so analysis-tool images are served.
        title: FastAPI app title.
        version: FastAPI app version.

    Returns:
        A configured FastAPI instance ready to be served by uvicorn.
    """
    catalog = model_catalog or DEFAULT_MODEL_CATALOG
    origins = cors_origins or CORS_ORIGINS

    app = FastAPI(title=title, version=version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health_api.make_router(catalog))
    app.include_router(models_api.make_router(catalog))
    app.include_router(chat_api.make_router(
        tools=tools, system_prompt=system_prompt, model_catalog=catalog,
    ))

    if plot_dir is not None:
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        app.mount("/plots", StaticFiles(directory=str(plot_path)), name="plots")

    return app
