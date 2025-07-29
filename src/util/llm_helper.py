from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import os


class LLMFactory:
    _instance = None
    _llm = None

    def __init__(self, config):
        if LLMFactory._instance is None:
            LLMFactory._instance = LLMFactory._initialize(config)

    @staticmethod
    def _initialize(config):
        server_type = config.LLM_SERVER_TYPE.lower()
        if server_type == "vllm" or server_type == "openai":
            llm = ChatOpenAI(
                api_key=os.environ.get("GEN_API_KEY", "EMPTY"),
                base_url=config.LLM_URL,
                model=config.GEN_MODEL_NAME,
                temperature=config.GEN_TEMPERATURE
            )
        elif server_type == "ollama":
            llm = Ollama(
                base_url=config.LLM_URL,
                model=config.GEN_MODEL_NAME,
                temperature=config.GEN_TEMPERATURE
            )
        elif server_type == "gemini":
            llm = ChatGoogleGenerativeAI(
                api_key=os.environ.get("GEN_API_KEY"),
                model=config.GEN_MODEL_NAME,
                temperature=config.GEN_TEMPERATURE
            )
        else:
            raise ValueError(f"Invalid LLM Server type {config.LLM_SERVER_TYPE}")
        LLMFactory._llm = llm
        return llm | LLMFactory.strip_thought

    @staticmethod
    def strip_thought(message: AIMessage):
        messages = message.content.split('</think>')
        thought = messages[0].replace('<think>', '').replace('</think>', '')
        message.content = messages[-1].strip("\n\n")
        message.response_metadata['thought'] = thought
        return message

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance = cls._initialize(config)
        return cls._instance
