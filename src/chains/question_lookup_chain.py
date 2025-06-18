import html

import config
import config as app_config
import json
from databases.qdrant import CustomQdrant
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    Runnable
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from operator import itemgetter
from models.user_question import Question
from qdrant_client import QdrantClient, async_qdrant_client
from util.chat_history_util import format_chat_history
from util.llm_helper import LLMFactory


class QuestionLookupChain:
    """Preforms vector matching over a database of similar questions and generates answers"""
    def __init__(self, config):
        q_client_async = async_qdrant_client.AsyncQdrantClient(url=config.QDRANT_URL)
        q_client_sync = QdrantClient(url=config.QDRANT_URL)
        self.embeddings = config.ollama_emb
        self.data_store = CustomQdrant(
            async_client=q_client_async,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embeddings=self.embeddings,
            client=q_client_sync
        )
        # initialize llm
        self.llm = LLMFactory(config=config)
        self.langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                        public_key=config.LANGFUSE_PUBLIC_KEY,
                                        host=config.LANGFUSE_HOST)

        #PROMPTS
        self.REPHRASE_PROMPT = self._get_prompt_from_langfuse("REPHRASE_PROMPT")
        self.ANSWER_GENERATION_PROMPT = self._create_answer_generation_prompt("ANSWER_GENERATION_PROMPT")

    def as_retrieval_chain(self, lookup_parameters=None) -> Runnable:
        # This branch checks history and rephrases the user query to maximize relevant retrival.
        # Eg, followup questions usually use pronouns and replacements making refrerence to older conversation peices,
        # This is trying to modify the user question mso it can be better used against the vector store which has
        # more direct questions.

        if lookup_parameters is None:
            lookup_parameters = {"k": 20}
        rephrase_branch = RunnableBranch(
            # check history
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                # if there is some history reformat it and rephrase it and pass it down the chain
                RunnablePassthrough.assign(
                    chat_history=lambda x: format_chat_history(x['chat_history'])
                ).with_config(
                    run_name="format_chat_history"
                )
                | self.REPHRASE_PROMPT
                | self.llm.with_config(name='rephrase_user_query')
                | StrOutputParser()
            ),
            # no chat history , pass the whole question
            RunnableLambda(itemgetter("input")),
        ).with_config(run_name="rephrase_based_on_chat")
        # so retrival chain looks like this , rephrase the last query as standalone question , and get some documents.
        retrival_chain = (rephrase_branch |
                            self.data_store.as_retriever(search_kwargs=lookup_parameters)
                                .with_config(run_name="qdrant_lookup") |
                            self._combine_documents)
        return retrival_chain.with_config(run_name='retrieve_documents')

    def as_generative_chain(self, lookup_parameters=None) -> Runnable:
        if lookup_parameters is None:
            lookup_parameters = {"k": 20}
        _inputs = RunnableParallel(
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                # by default the context will contain top 20 documents
                "context": self.as_retrieval_chain(lookup_parameters=lookup_parameters)
                            .with_config(run_name="retrival"),
            }
        ).with_types(input_type=Question).with_config(run_name="question_lookup_chain")

        answer_chain = ((self.ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser())
                        .with_config(name='answer_generation'))
        generative_chain = (_inputs | RunnableParallel(
            {
                "output":  RunnableLambda(lambda x: {
                    "input": x["input"],
                    "context": x.get("context", ""),
                    "chat_history": x["chat_history"]}
                    ) | answer_chain,
                "extra": {}
            }
        ))
        return config.configure_langfuse(generative_chain)

    def _get_raw_from_langfuse(self, prompt_name:str)-> str:
        """Gets raw string for of prompts in langfuse"""
        return self.langfuse_client.get_prompt(prompt_name).prompt

    def _get_prompt_from_langfuse(self, prompt_name: str)-> PromptTemplate:
        """Constructs langchain prompt object by getting raw string from langfuse"""
        return PromptTemplate.from_template(template=self._get_raw_from_langfuse(prompt_name))

    def _create_answer_generation_prompt(self, prompt_name: str) -> ChatPromptTemplate:
        """Creates answer generation prompt"""
        template = self._get_raw_from_langfuse(prompt_name)
        return ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

    @staticmethod
    def _combine_documents(docs, document_separator="\n\n", *args, **kwargs):
        docs_seen = []
        doc_strings = []
        document_prompt = PromptTemplate.from_template(template="{page_content}")
        for document in docs:
            if document.metadata['study_id'] not in docs_seen:
                if not document.page_content:
                    continue
                document.page_content = json.loads(document.page_content)
                raw_page_content = document.page_content['abstract']
                safe_page_content = html.escape(raw_page_content)

                doc_strings.append(f'<study id="{document.metadata['study_id']}">'
                                   f'<title>{document.page_content['title']}</title>'
                                   f'<abstract>{safe_page_content}</abstract></study>')
                docs_seen.append(document.metadata['study_id'])
        joined_docs = document_separator.join(doc_strings)
        return f"<studies>{joined_docs}</studies>"


# To test run this code as main...
if __name__ == '__main__':
    cls = QuestionLookupChain(config=app_config)
    user_q = Question(chat_history=[], input="What was the purpose of Genome-wide Association Study of Adiposity in Samoans?")
    qa_chain = cls.as_generative_chain()
    response = qa_chain.invoke(user_q.dict())
    print(response)