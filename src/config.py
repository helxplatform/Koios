from langfuse import Langfuse
from langchain_community.embeddings import OllamaEmbeddings
import os
from pathlib import Path
LLM_URL = os.getenv('LLM_URL', 'https://vllm.apps.renci.org/v1').rstrip('/')
EMBEDDING_URL = os.getenv('EMBEDDING_URL', 'http://localhost:11434')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333').rstrip('/')
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
GUARDIAN_MODEL_NAME = os.getenv("GUARDIAN_MODEL_NAME", "llama3.1:latest")
GUARDIAN_MODEL_HOST = os.getenv("GUARDIAN_MODEL_URL", "http://localhost:11434")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "Losspost/stella_en_1.5b_v5")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "questions_on_study_abstract_stella")
STUDIES_JSON_FILE = os.getenv("STUDIES_JSON_FILE", Path(os.path.dirname(__file__), '..',  'data', '99_studies.json'))
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", Path(os.path.dirname(__file__), '..' ,  'koios.log'))
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LLM_SERVER_TYPE = os.getenv("LLM_SERVER_TYPE", "VLLM")
ollama_emb = OllamaEmbeddings(model=EMB_MODEL_NAME, base_url=EMBEDDING_URL)
ollama_emb.query_instruction = ""
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_GRAPH_NAME = os.getenv("REDIS_GRAPH_NAME", "")
TMP_DIR = os.getenv("TMP_DIR", os.path.join(os.path.dirname(os.path.realpath(__file__)),'..', 'tmp'))



if LANGFUSE_ENABLED:
    langfuse = Langfuse(secret_key=LANGFUSE_SECRET_KEY,
                        public_key=LANGFUSE_PUBLIC_KEY,
                        host=LANGFUSE_HOST)
else:
    langfuse = None


def configure_langfuse(runnable):
    if LANGFUSE_ENABLED:
        from langchain_core.runnables.config import RunnableConfig
        from langfuse.callback import CallbackHandler

        langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        langfuse_handler.auth_check()
        runnable_config = RunnableConfig(callbacks=[langfuse_handler])
        return runnable.with_config(runnable_config)
    return runnable