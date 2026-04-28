"""ai_core — reusable FastAPI AI assistant: providers, tool loop, RAG, SSE streaming.

Public API:
  create_app(...)          → FastAPI app with /ai/* routes
  BaseTool                  → subclass to add project-specific tools
  BaseProvider              → subclass for new model providers
  SearchDocsTool            → generic RAG-backed tool (constructed with knowledge_dir)
"""
from ai_core.app_factory import create_app
from ai_core.providers.base import BaseProvider
from ai_core.schemas.chat import ChatRequest, ChatResponse, Message
from ai_core.schemas.provider import ProviderResponse
from ai_core.schemas.tool import ToolCall, ToolResult
from ai_core.tools.base import BaseTool
from ai_core.tools.search_docs import SearchDocsTool

__all__ = [
    "create_app",
    "BaseProvider",
    "BaseTool",
    "SearchDocsTool",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "ProviderResponse",
    "ToolCall",
    "ToolResult",
]
