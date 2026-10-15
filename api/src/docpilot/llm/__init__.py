from docpilot.config import Settings
from docpilot.llm.base import LLMProvider


def make_provider(settings: Settings) -> LLMProvider:
    """Pick the chat model backend from LLM_PROVIDER. Imports are lazy to keep startup light."""
    if settings.llm_provider == "fake":
        from docpilot.llm.fake_provider import FakeProvider

        return FakeProvider()
    if settings.llm_provider == "openai":
        from docpilot.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    from docpilot.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(settings)
