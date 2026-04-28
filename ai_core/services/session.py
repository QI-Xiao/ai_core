"""In-memory conversation history keyed by session_id. Lost on server restart."""
from ai_core.schemas.chat import Message

_sessions: dict[str, list[Message]] = {}


def get_history(session_id: str) -> list[Message]:
    return list(_sessions.get(session_id, []))


def append_messages(session_id: str, messages: list[Message]) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].extend(messages)


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
