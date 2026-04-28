from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = {}
