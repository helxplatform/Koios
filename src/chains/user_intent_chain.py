from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import asyncio

import config
from util.study_data import get_study_data
from util.chat_history_util import format_chat_history
from models.user_question import Question

import config as app_config
from util.llm_helper import LLMFactory
from langfuse import Langfuse


class UserIntentChain:
    def __init__(self, config):
        self.config = config
        self.llm: BaseChatModel = LLMFactory.get_llm(config=config)

        self.langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                        public_key=config.LANGFUSE_PUBLIC_KEY,
                                        host=config.LANGFUSE_HOST)

        # prompts
        self.USER_INTENT_PROMPT = self._get_system_prompt("INTENT_PROMPT")

    ######
    #  Begin Chain definitions
    ####
    def as_intent_extraction_chain(self):
        return (self.USER_INTENT_PROMPT |
                self.llm.with_config(name='user_intent_generation') |
                JsonOutputParser())

    def as_generative_chain(self):
        extraction_chain = self.as_intent_extraction_chain().with_config(run_name="user_intent")
        generative_chain = extraction_chain | RunnableLambda(
            lambda x: {
                "user_intent": {
                    "as_json": x,
                    "as_prompt": self.__format_as_prompt_snippet(x)
                }
            }
        )
        return config.configure_langfuse(generative_chain.with_config(run_name="user_intent_generation"))

    @staticmethod
    def __format_as_prompt_snippet(generation_output: dict):
        return "\n\t - ".join(
            [f"The user's {k.replace('_', ' ')} : {v}" for k, v in generation_output.items()])

    ######
    #  // end chain definitions
    ####

    ######
    #  Begin langfuse interactions (prompt definitions)
    ####
    def _get_raw_from_langfuse(self, prompt_name: str) -> str:
        """Gets raw string for of prompts in langfuse"""
        return self.langfuse_client.get_prompt(prompt_name).prompt

    def _get_prompt_from_langfuse(self, prompt_name: str) -> PromptTemplate:
        """Constructs langchain prompt object by getting raw string from langfuse"""
        return PromptTemplate.from_template(template=self._get_raw_from_langfuse(prompt_name))

    def _get_system_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        """Constructs concept extraction prompt object"""
        return ChatPromptTemplate.from_messages(
            ["system", self._get_raw_from_langfuse(prompt_name)]
        )

    def _create_answer_generation_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        template = self._get_raw_from_langfuse(prompt_name)
        return ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )

    ######
    #  End langfuse interactions
    ####


if __name__ == "__main__":
    kg_agent = UserIntentChain(config=app_config)
    # user_q = Question(chat_history=[], input="what studies are there about sickle cell?")
    user_q = Question(chat_history=[], input="please find studies related to sickle cell")
    qa_chain = kg_agent.as_generative_chain()
    response = asyncio.run(qa_chain.ainvoke(user_q.dict()))
    import json

    print(json.dumps(response, indent=2))
