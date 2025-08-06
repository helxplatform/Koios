from doctest import run_docstring_examples
from fastapi import FastAPI
import config
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from models.user_question import Question
from typing import Dict, List
from langchain_core.prompts.prompt import PromptTemplate
import xml.etree.ElementTree as ET
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse import Langfuse
from langchain_core.output_parsers import StrOutputParser
from guardrails.input_guard import InputGuard
from util.chat_history_util import format_chat_history
from util.llm_helper import LLMFactory

class QVKGChain:
    def __init__(self, config, default_k=10):
        self.config = config
        self.kg_chain = KGChain(config=config)
        self.qv_chain = QuestionLookupChain(config=config)
        self.llm = LLMFactory(config=config)
        self.default_lookup_k = default_k
        self.langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                        public_key=config.LANGFUSE_PUBLIC_KEY,
                                        host=config.LANGFUSE_HOST)

        self.COMBINED_ANSWER_PROMPT = self._create_answer_generation_prompt("QV_KG_PROMPT")

    def as_retrieval_chain(self, lookup_parameters=None):
        """Chain to retrieve data from both sources in parallel"""
        if lookup_parameters is None:
            lookup_parameters = {"k": self.default_lookup_k}
        
        return RunnableParallel({
            "kg_retrieval": self.kg_chain.as_retrival_chain().with_config(run_name="kg_retrieval"),
            "qv_retrieval": self.qv_chain.as_retrieval_chain(lookup_parameters=lookup_parameters).with_config(run_name="qv_retrieval"),
            "input": lambda x: x["input"],
            "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
            "user_intent": lambda x: x.get("user_intent", {})
        })

    @staticmethod
    def combine_xml_outputs(
            kg_context: str,
            qv_context: str,
            separator: str = "\n\n"
    ) -> str:
        """
        Combines two well-formed XML strings containing study data,
        prioritizing the rich_xml_str and deduplicating by study_id.
        """
        study_data: Dict[str, str] = {}

        # --- 1. Process the rich content string first ---
        if kg_context.strip():
            # No pre-processing needed! We can parse it directly.
            root = ET.fromstring(kg_context)
            for study_element in root.findall('study'):
                study_id = study_element.get('id')
                if study_id:
                    study_block = ET.tostring(study_element, encoding='unicode').strip()
                    study_data[study_id] = study_block

        # --- 2. Process the simple content string ---
        if qv_context.strip():
            # No pre-processing needed here either.
            root = ET.fromstring(qv_context)
            for study_element in root.findall('study'):
                study_id = study_element.get('id')
                # Only add if this study_id has not been seen before
                if study_id and study_id not in study_data:
                    study_block = ET.tostring(study_element, encoding='unicode').strip()
                    study_data[study_id] = study_block

        # --- 3. Join the unique study blocks into a final string ---
        combined_docs: List[str] = list(study_data.values())
        return "<studies>" + separator.join(combined_docs) + "</studies>"


    def as_generative_chain(self, lookup_parameters=None):
        """Chain to combine results and generate an answer"""
        if lookup_parameters is None:
            lookup_parameters = {"k": self.default_lookup_k}
        
        retrieval_chain = self.as_retrieval_chain(lookup_parameters)

        combined_chain_data = RunnableLambda(lambda x: {
            "input": x["input"],
            "context": self.combine_xml_outputs(kg_context=x.get("kg_retrieval", {}).get("context", {})
                                                .get("context", "<studies></studies>") or "<studies></studies>",
                                                qv_context=x.get("qv_retrieval") or "<studies></studies>"),
            "chat_history": x["chat_history"],
            "has_context": bool(x.get("kg_retrieval", {}).get("context", {}).get("context", ""))
                           or bool(x.get("qv_retrieval", "")),
            "kg_extra": x.get("kg_retrieval", {}).get("context", {}).get("extra_data", {}),
            "user_persona": x.get("user_intent", {}).get("as_prompt", "")
        }).with_config(run_name="combined_chain_data")

        response_branch = RunnableBranch(
            (
                lambda x: x["has_context"],
                RunnableParallel({
                    "output": RunnableLambda(lambda x: {
                        "input": x["input"],
                        "context": x["context"],
                        "chat_history": x["chat_history"],
                        "user_persona": x["user_persona"]
                    }) | self.COMBINED_ANSWER_PROMPT | self.llm.with_config(name="answer_generation") | StrOutputParser(),
                    "extra": RunnableLambda(lambda x: {"knowledge_graph": x["kg_extra"].get("knowledge_graph", {})}),
                    "prompt": RunnableLambda(lambda x: {
                        "input": x["input"],
                        "context": x["context"],
                        "chat_history": x["chat_history"],
                        "user_persona": x["user_persona"]
                    }) | self.COMBINED_ANSWER_PROMPT
                })
            ),
            RunnableLambda(lambda x: {
                "output": "No studies or relevant information were found to answer the query.",
                "extra": {},
                "prompt": RunnableLambda(lambda x: {
                    "input": x["input"],
                    "context": x["context"],
                    "chat_history": x["chat_history"],
                    "user_persona": x["user_persona"]
                }) | self.COMBINED_ANSWER_PROMPT
            })
        ).with_config(run_name="response_branch")
        qvkg_chain = retrieval_chain | combined_chain_data | response_branch

        return config.configure_langfuse(qvkg_chain.with_config(run_name="qvkg_lookup_generation"))

    def _create_answer_generation_prompt(self, prompt_name):
        user_prompt = self._get_raw_from_langfuse(prompt_name)
        return ChatPromptTemplate([
            ("user", user_prompt),
            # @TODO test out chain of draft
            # Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.
            # Return the answer at the end of the response after the following separator  `$!`.
            #
            # """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
    def _get_raw_from_langfuse(self, prompt_name: str) -> str:
        """Gets raw string for of prompts in langfuse"""
        return self.langfuse_client.get_prompt(prompt_name).prompt



if __name__ == "__main__":
    import asyncio
    import json
    from models.user_question import Question
    import config as app_config
    import config
    from chains.qvkg_chain import QVKGChain

    qvkg_agent = QVKGChain(config=app_config)
    user_q = Question(chat_history=[("Saliva","This study focuses on whole-genome sequencing and related phenotypes in asthma. It includes variables related to demographic details, health status, and genetic data. For example, it collects data on age, gender, and other phenotypic characteristics that may be linked to secretory status.")], input="variables related to Saliva secretor studies")
    qa_chain = qvkg_agent.as_generative_chain()
    response = asyncio.run(qa_chain.ainvoke(user_q.dict()))

    print(json.dumps(response, indent=2))