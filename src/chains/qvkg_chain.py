from doctest import run_docstring_examples
from fastapi import FastAPI
import config
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from models.user_question import Question
from typing import Dict, List, Optional
from langchain_core.prompts.prompt import PromptTemplate
import xml.etree.ElementTree as ET
import re
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
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
        self.raw_llm = LLMFactory.get_raw_llm(config=config)
        self.llm = self.raw_llm | LLMFactory.strip_thought
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
    def _safe_parse_xml(xml_string: str) -> Optional[ET.Element]:
        """
        Helper: Sanitizes string and attempts to parse XML safely.
        Handles unescaped chars and missing root elements.
        """
        if not xml_string or not xml_string.strip():
            return None

        # 1. Sanitize: Remove illegal XML control characters
        # (characters like vertical tabs that LLMs sometimes hallucinate)
        xml_string = re.sub(
            u'[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]+', 
            '', 
            xml_string
        )

        # 2. Sanitize: Fix unescaped ampersands (e.g., "R&D" -> "R&amp;D")
        # Looks for & NOT followed by valid entity or hash
        xml_string = re.sub(r'&(?!(?:amp|lt|gt|apos|quot|#\d+|#x[0-9a-fA-F]+);)', '&amp;', xml_string)

        try:
            # Attempt 1: Parse directly
            return ET.fromstring(xml_string)
        except ET.ParseError:
            try:
                # Attempt 2: It might lack a single root element. Wrap it.
                # e.g., input was "<study>...</study><study>...</study>"
                return ET.fromstring(f"<root>{xml_string}</root>")
            except ET.ParseError:
                # If it still fails, the XML is too broken to save without 'lxml'
                print(f"Warning: Failed to parse XML content. Skipping block.")
                return None

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
            root = QVKGChain._safe_parse_xml(kg_context)
            for study_element in root.findall('study'):
                study_id = study_element.get('id')
                if study_id:
                    study_block = ET.tostring(study_element, encoding='unicode').strip()
                    study_data[study_id] = study_block

        # --- 2. Process the simple content string ---
        if qv_context.strip():            
            root = QVKGChain._safe_parse_xml(qv_context)
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
                    }) | RunnableLambda(self._limit_context_tokens) | self.COMBINED_ANSWER_PROMPT | self.llm.with_config(name="answer_generation") | StrOutputParser() | RunnableLambda(lambda content: AIMessage(content=content, name="qvkg_chain")),
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
                "output": AIMessage(content="No studies or relevant information were found to answer the query.", name="qvkg_chain"),
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

    def _limit_context_tokens(self, inputs: Dict) -> Dict:
        """
        Checks the token count of the prompt constructed from inputs.
        If it exceeds the limit, iteratively removes variables from studies, then studies themselves until it fits.
        """
        MAX_TOKENS = 16000  # Leave some buffer for response

        # We need to format the messages to count tokens accurately
        def get_token_count(current_inputs):
            messages = self.COMBINED_ANSWER_PROMPT.format_messages(**current_inputs)
            try:
                return self.raw_llm.get_num_tokens_from_messages(messages)
            except Exception:
                # Fallback if the LLM doesn't support token counting
                # Rough estimate: 4 chars per token
                return sum(len(m.content) for m in messages) / 4

        if get_token_count(inputs) <= MAX_TOKENS:
            return inputs

        # Parse context
        try:
            context_str = inputs["context"]
            # combine_xml_outputs returns <studies>...</studies>
            root = ET.fromstring(context_str)
            
            # Loop until we are under the limit
            while get_token_count(inputs) > MAX_TOKENS:
                studies = root.findall('study')
                if not studies:
                    break
                
                # Check if we have any variables to remove
                # We want to find the study with the most variables
                study_with_most_vars = None
                max_vars = 0
                
                for study in studies:
                    variables_container = study.find('variables')
                    if variables_container is not None:
                        vars_in_study = variables_container.findall('variable')
                        if len(vars_in_study) > max_vars:
                            max_vars = len(vars_in_study)
                            study_with_most_vars = study
                
                # If we found a study with variables (and count > 0), remove one
                if study_with_most_vars is not None and max_vars > 0:
                    variables_container = study_with_most_vars.find('variables')
                    vars_in_study = variables_container.findall('variable')
                    # Remove the last variable
                    variables_container.remove(vars_in_study[-1])
                else:
                    # No variables left in any study, remove the last study
                    root.remove(studies[-1])
                
                # Reconstruct context
                new_context = ET.tostring(root, encoding='unicode')
                inputs["context"] = new_context

        except ET.ParseError:
            # If we can't parse, we can't intelligently reduce.
            pass

        return inputs


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