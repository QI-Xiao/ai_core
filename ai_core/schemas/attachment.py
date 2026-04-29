"""Typed attachment shared across the chat pipeline.

A tool that produces side-effect output (a rendered image, a structured
dataset, …) returns one or more `Attachment` objects from
`BaseTool.run_with_attachments`. The streaming chat service forwards each
attachment to the SSE stream as `{"type": "attachment", "kind", "url", "data"}`.

`ai_core` never inspects `kind`: it is an opaque consumer-defined string.
That keeps the library project-agnostic — new attachment kinds can be added
in consumers without touching `ai_core`.
"""
from typing import Any

from pydantic import BaseModel


class Attachment(BaseModel):
    """A single side-effect produced by a tool call.

    Either `url` (for images and other addressable resources) or `data`
    (for structured payloads to render client-side) — or both — should be
    populated, depending on what the consumer's frontend knows how to render.
    """

    kind: str
    url: str | None = None
    data: dict[str, Any] | None = None
