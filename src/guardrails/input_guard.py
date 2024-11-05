from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails import RailsConfig
import os
import tempfile
from langfuse import Langfuse
from util.llm_helper import LLMFactory


class InputGuard:
    _instance = None

    def __init__(self, config):
        if InputGuard._instance is None:
            InputGuard._instance = InputGuard._initialize(config)

    @staticmethod
    def _initialize(config):
        temp_dir_root = config.TMP_DIR
        langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                   public_key=config.LANGFUSE_PUBLIC_KEY,
                                   host=config.LANGFUSE_HOST)
        rails_config = config.langfuse.get_prompt("GUARDRAILS_CONFIG").prompt
        rails_prompt = config.langfuse.get_prompt("INPUT_GUARDRAILS").prompt
        rail_config_dir = InputGuard.setup_config_dir(temp_dir_root,
                                                      rails_config=rails_config,
                                                      rails_prompt=rails_prompt
                                                      )
        rails_config = RailsConfig.from_path(rail_config_dir)
        llm = LLMFactory(config)
        instance = RunnableRails(rails_config, llm)
        return instance

    @staticmethod
    def setup_config_dir(dir_root, rails_config, rails_prompt):
        temp_dir = tempfile.mkdtemp(dir=dir_root)
        guard_config_path = os.path.join(temp_dir, "config")
        os.makedirs(guard_config_path, exist_ok=True)
        config_file_path = os.path.join(guard_config_path, "config.yml")
        prompt_file_path = os.path.join(guard_config_path, "prompt.yml")
        with open(config_file_path, 'w') as config_file:
            config_file.write(rails_config)
        with open(prompt_file_path, 'w') as prompt_file:
            prompt_file.write(rails_prompt)
        return guard_config_path

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance = cls._initialize(config)
        return cls._instance

