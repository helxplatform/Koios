from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import asyncio

from models.user_question import Question
import config as app_config
from util.llm_helper import LLMFactory
from langfuse import Langfuse


class QueryRelevanceClassifierChain:
    def __init__(self, config):
        self.config = config
        self.llm: BaseChatModel = LLMFactory.get_llm(config=config)
        self.langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                        public_key=config.LANGFUSE_PUBLIC_KEY,
                                        host=config.LANGFUSE_HOST)

        # prompts
        self.PROMPT = self._get_system_prompt("QUERY_RELEVANCE_CLASSIFIER_PROMPT")

    ######
    #  Begin Chain definitions
    ####
    def as_routing_chain(self):
        run_config = RunnableConfig(
            run_name="query_relevance_classifier_generation",
        )

        return (self.PROMPT |
                self.llm.with_config(run_config) |
                StrOutputParser())

    def as_generative_chain(self):
        extraction_chain = self.as_routing_chain().with_config(run_name="query_classification")
        generative_chain = extraction_chain | RunnableLambda(
            lambda x: {
                "next": x
            }
        )
        return app_config.configure_langfuse(generative_chain.with_config(run_name="query_classification_generation"))

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

    ######
    #  End langfuse interactions
    ####


if __name__ == "__main__":
    kg_agent = QueryRelevanceClassifierChain(config=app_config)
    # user_q = Question(chat_history=[], input="what studies are there about sickle cell?")
    user_q = Question(chat_history=[], input="what can this chat bot do?")
    qa_chain = kg_agent.as_generative_chain()
    response = asyncio.run(qa_chain.ainvoke(user_q.model_dump()))
    import json

    print(json.dumps(response, indent=2))
