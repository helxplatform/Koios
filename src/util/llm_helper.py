from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from typing import Callable


class LLMFactory:

    @staticmethod
    def get_raw_llm(config):
        llm_raw = None
        server_type = config.LLM_SERVER_TYPE.lower()
        if server_type in ("vllm", "openai"):
            llm_raw = ChatOpenAI(
                api_key=os.environ.get("GEN_API_KEY", "EMPTY"),
                base_url=config.LLM_URL,
                model=config.GEN_MODEL_NAME,
                temperature=config.GEN_TEMPERATURE,
            )
        elif server_type == "ollama":
            llm_raw = Ollama(
                base_url=config.LLM_URL,
                model=config.GEN_MODEL_NAME,
                temperature=config.GEN_TEMPERATURE,
            )
        elif server_type == "gemini":
            # create only when first requested inside the active loop
            llm_raw = ChatGoogleGenerativeAI(
                api_key=os.environ.get("GEN_API_KEY"),
                model=config.GEN_MODEL_NAME,
                temperature=config.GEN_TEMPERATURE,
            )
        else:
            raise ValueError(f"Invalid LLM Server type {config.LLM_SERVER_TYPE}")
        return llm_raw

    @classmethod
    def get_llm(cls, config):
        llm_raw = LLMFactory.get_raw_llm(config)
        _llm = llm_raw | LLMFactory.strip_thought
        return _llm

    @staticmethod
    def strip_thought(message: AIMessage):
        messages = message.content.split('</think>')
        thought = messages[0].replace('<think>', '').replace('</think>', '')
        message.content = messages[-1].strip("\n\n")
        message.response_metadata['thought'] = thought
        return message


class DeferredLLM:
    """
    Lazily creates a concrete LLM instance when any attribute/method is first accessed.
    The creation happens in the thread (and event loop) that triggers the first access.
    `factory` should be a callable that takes no args and returns a concrete LLM instance.
    """
    def __init__(self, factory: Callable[[], object]):
        self._factory = factory
        self._llm = None

    def _ensure(self):
        if self._llm is None:
            self._llm = self._factory()

    def __getattr__(self, item):
        # called from the thread/method that uses the llm
        self._ensure()
        return getattr(self._llm, item)
