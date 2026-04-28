from ai_core.config import AI_DEFAULT_MODEL, ANTHROPIC_DEFAULT_MODEL
from ai_core.providers.base import BaseProvider

# Default model catalog. Projects can pass their own to create_app(model_catalog=...).
DEFAULT_MODEL_CATALOG: list[dict] = [
    {"id": "claude-opus-4-7",          "display_name": "Claude Opus 4.7",   "provider": "anthropic", "default": False},
    {"id": "claude-sonnet-4-6",        "display_name": "Claude Sonnet 4.6", "provider": "anthropic", "default": True},
    {"id": "claude-haiku-4-5-20251001","display_name": "Claude Haiku 4.5",  "provider": "anthropic", "default": False},
    {"id": "gpt-4o",                   "display_name": "GPT-4o",            "provider": "openai",    "default": False},
    {"id": "gpt-4o-mini",              "display_name": "GPT-4o mini",       "provider": "openai",    "default": False},
]


def get_provider(
    model: str | None = None,
    *,
    catalog: list[dict] | None = None,
) -> BaseProvider:
    """Return a provider instance for the given model ID.

    Falls back to AI_DEFAULT_MODEL or ANTHROPIC_DEFAULT_MODEL when model is None/empty.
    """
    cat = catalog or DEFAULT_MODEL_CATALOG
    model_to_provider = {m["id"]: m["provider"] for m in cat}
    default_model = AI_DEFAULT_MODEL or ANTHROPIC_DEFAULT_MODEL
    resolved = model or default_model

    provider_name = model_to_provider.get(resolved)
    if provider_name is None:
        raise ValueError(
            f"Unknown model: {resolved!r}. "
            f"Valid models: {[m['id'] for m in cat]}"
        )

    if provider_name == "anthropic":
        from ai_core.providers.anthropic import AnthropicProvider
        return AnthropicProvider(model=resolved)
    if provider_name == "openai":
        from ai_core.providers.openai import OpenAIProvider
        return OpenAIProvider(model=resolved)
    if provider_name == "ollama":
        from ai_core.providers.ollama import OllamaProvider
        return OllamaProvider()
    if provider_name == "vllm":
        from ai_core.providers.vllm import VLLMProvider
        return VLLMProvider()

    raise ValueError(f"Unhandled provider: {provider_name!r}")
