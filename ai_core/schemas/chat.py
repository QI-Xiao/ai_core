from typing import Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: Optional[str] = None
    model: Optional[str] = None  # model ID; falls back to server default


class ChatResponse(BaseModel):
    message: Message
    session_id: str
    model: str
