from doctest import run_docstring_examples
from fastapi import FastAPI
import config
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from models.user_question import Question
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        
        # Hardcoded combined answer generation prompt
        self.COMBINED_ANSWER_PROMPT = ChatPromptTemplate([
            ("user", """You are a biomedical expert tasked with answering questions about scientific studies using the provided information.
You have two sources of information:
1. Knowledge Graph Data: This outlines relationships between biomedical concepts and variables, which may be linked to the same or different studies.
2. Study abstract Data: This contains textual descriptions and contextual information about the studies themselves.
Use both sources to provide a comprehensive answer that can satisfy the user query. When the sources provide complementary 
information, combine them. When they provide contradictory information, as an expert share the source might be more reliable for this specific query.

Knowledge Graph Context:
{kg_context}

Study abstract Context:
{qv_context}

Always cite specific studies with their IDs when they appear in your answer. Here is what we know about the user asking the question:
<user_persona>
    {user_persona}
</user_persona>
"""),
    # @TODO test out chain of draft
    # Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.
    # Return the answer at the end of the response after the following separator  `$!`.
    #
    # """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
    
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

    def as_generative_chain(self, lookup_parameters=None):
        """Chain to combine results and generate an answer"""
        if lookup_parameters is None:
            lookup_parameters = {"k": self.default_lookup_k}
        
        retrieval_chain = self.as_retrieval_chain(lookup_parameters)

        combined_chain_data = RunnableLambda(lambda x: {
            "input": x["input"],
            "kg_context": x.get("kg_retrieval", {}).get("context", {}).get("context", ""),
            "qv_context": x.get("qv_retrieval", ""),
            "chat_history": x["chat_history"],
            "has_context": bool(x.get("kg_retrieval", {}).get("context", {}).get("context", "")) or bool(x.get("qv_retrieval", "")),
            "kg_extra": x.get("kg_retrieval", {}).get("context", {}).get("extra_data", {}),
            "user_persona": x.get("user_intent", {}).get("as_prompt", "")
        }).with_config(run_name="combined_chain_data")

        response_branch = RunnableBranch(
            (
                lambda x: x["has_context"],
                RunnableParallel({
                    "output": RunnableLambda(lambda x: {
                        "input": x["input"],
                        "kg_context": x["kg_context"],
                        "qv_context": x["qv_context"],
                        "chat_history": x["chat_history"],
                        "user_persona": x["user_persona"]
                    }) | self.COMBINED_ANSWER_PROMPT | self.llm.with_config(name="answer_generation") | StrOutputParser(),
                    "extra": RunnableLambda(lambda x: {"knowledge_graph": x["kg_extra"]})

                })
            ),
            RunnableLambda(lambda x: {
                "output": "No studies or relevant information were found to answer the query.",
                "extra": {}
            })
        ).with_config(run_name="response_branch")
        qvkg_chain = retrieval_chain | combined_chain_data | response_branch

        return config.configure_langfuse(qvkg_chain.with_config(run_name="qvkg_lookup_generation"))
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