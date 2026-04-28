from typing import Any, Optional

from pydantic import BaseModel

from ai_core.schemas.tool import ToolCall


class ProviderResponse(BaseModel):
    """Provider-agnostic response. Never expose SDK-specific types above this layer."""
    content: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    tool_calls: list[ToolCall] = []
    raw_content: list[dict[str, Any]] = []
