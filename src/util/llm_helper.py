from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import os


class LLMFactory:
    _instance = None

    def __init__(self, config):
        if LLMFactory._instance is None:
            LLMFactory._instance = LLMFactory._initialize(config)

    @staticmethod
    def _initialize(config):
        server_type = config.LLM_SERVER_TYPE.lower()
        if server_type == "vllm" or server_type == "openai":
            llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_KEY", "EMPTY"),
                base_url=config.LLM_URL,
                model=config.GEN_MODEL_NAME
            )
        elif server_type == "ollama":
            llm = Ollama(
                base_url=config.LLM_URL,
                model=config.GEN_MODEL_NAME
            )
        else:
            raise ValueError(f"Invalid LLM Server type {config.LLM_SERVER_TYPE}")
        return llm

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance = cls._initialize(config)
        return cls._instance
