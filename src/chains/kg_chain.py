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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import asyncio

import config
from util.study_data import get_study_data
from util.chat_history_util import format_chat_history
from models.user_question import Question

import config as app_config
import pandas as pd
from util.llm_helper import LLMFactory
from langfuse import Langfuse
from databases.redis_graph import RedisGraphDB
from util.ner_utils import resolve_to_identifiers_sapbert


class KGChain:
    def __init__(self, config):
        self.config = config
        self.llm: BaseChatModel = LLMFactory(config=config)

        self.langfuse_client = Langfuse(secret_key=config.LANGFUSE_SECRET_KEY,
                                        public_key=config.LANGFUSE_PUBLIC_KEY,
                                        host=config.LANGFUSE_HOST)

        # prompts
        self.CONCEPT_EXTRACTION_PROMPT = self._get_system_prompt("CONCEPT_EXTRACTION_PROMPT")
        self.ANSWER_GENERATION_PROMPT = self._create_answer_generation_prompt("ANSWER_GENERATION_PROMPT_KG_APP")
        self.graph = RedisGraphDB(config)

    ######
    #  Begin Chain definitions
    ####
    def as_concept_extraction_chain(self):
        return (self.CONCEPT_EXTRACTION_PROMPT | self.llm | StrOutputParser()).with_config(
            run_name="concept_extraction")

    def as_retrival_chain(self):
        return RunnableParallel(
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "context": (self.as_concept_extraction_chain() | self._get_studies_as_runnable() | StrOutputParser())
                .with_config(run_name="retrival")
            }
        )

    def as_generative_chain(self):
        retrival_chain = self.as_retrival_chain()
        answer_chain = RunnableBranch(
            # check if we can get some studies from the graph.
            (
                RunnableLambda(lambda x: bool(x.get("context"))).with_config(
                    run_name="has_context"
                ),
                (self.ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser()).with_config(run_name="answer_generation")
            ),
            # If no studies from the graph, and empty context respond with static text
            RunnableLambda(lambda x: "No studies were found to answer the query.").with_config(run_name="no_data"),
        )
        generative_chain = retrival_chain | answer_chain
        return config.configure_langfuse(generative_chain)

    ######
    #  // end chain definitions
    ####

    ######
    #  Begin Study retrieval method definitions
    ####

    async def _get_one_hop_variables(self, concept_id, limit=100):
        """
        Gets variables one hop away from a concept
        :return:
        """
        cypher_query = f"""
            MATCH (c {{id: "{concept_id}"}})-[r1]->(v:`biolink.StudyVariable`)-[r2]->(s:`biolink.Study`)
            RETURN c.name AS concept_name,v.name AS variable_name,v.id AS variable_id,v.description AS variable_desc, s.id AS study_id
            LIMIT {limit}
        """
        result = await self.graph.query_graph(cypher_query)
        if result.result_set is None:
            return pd.DataFrame(columns=['concept_name', 'variable_name', 'variable_id', 'variable_desc', 'study_id'])
        data = result.result_set
        df = pd.DataFrame(data, columns=['concept_name', 'variable_name', 'variable_id', 'variable_desc', 'study_id'])
        df['study_id'] = df['study_id'].apply(lambda x: x.split('.')[0])
        return df

    async def _get_study_documents(self, concept_strings):
        """ This function is where the core of the retrival from kg is"""
        # use sap-bert to resolve the concepts
        concepts = await resolve_to_identifiers_sapbert(concept_strings.split(','))

        if not concepts:
            return "No information available"

        # do concurrent requests to the graph for max speed.
        lookup_tasks = [self._get_one_hop_variables(concept_curie)
                        for concept_label, concept_curie in concepts.items()]

        data_frames = await asyncio.gather(*lookup_tasks)

        # merge dataframes into a single table
        final_data_frame = pd.concat(data_frames, ignore_index=True)

        summary_data_frame = final_data_frame.groupby('study_id').agg(
            concept_list=('concept_name', lambda x: ', '.join(x)),
            variable_desc_list=('variable_desc', lambda x: ', '.join(x)),
            variable_name_list=('variable_name', lambda x: ', '.join(x)),
            variable_id_list=('variable_id', lambda x: ', '.join(x)),
            number_of_concepts=('concept_name', 'nunique')
        ).reset_index()

        # sort the frame with concept number
        summary_data_frame = summary_data_frame.sort_values(by='number_of_concepts', ascending=False)

        # merge variable as strings (?)
        summary_data_frame['variable_id_list'] = summary_data_frame['variable_id_list'].apply(
            lambda x: ', '.join(set([vid.split('.')[0] for vid in x.split(', ')]))
        )

        # add study description to the data frame
        summary_data_frame[['study_name', 'permalink', 'description']] = summary_data_frame['study_id'].apply(
            lambda x: pd.Series(
                # this is just using the file but instead of matching on full study id its using the first part.
                get_study_data(x,
                               lambda in_file, current_study_id:
                               in_file.split('.')[0] == current_study_id.split('.')[0],
                               exclude_keys=["study_id"]
                               )
            ))

        # filter empty study names,
        summary_data_frame = summary_data_frame[summary_data_frame['study_name'] != ""]

        # Concatenate the `variable_name`, `variable_id`, and `variable_desc`
        summary_data_frame['variable_info'] = summary_data_frame.apply(
            lambda row: '\n'.join([f"\t {name} ({var_id}): {desc}"
                                   for name, var_id, desc in zip(
                    row['variable_name_list'].split(', '),
                    row['variable_id_list'].split(', '),
                    row['variable_desc_list'].split(', ')
                )]), axis=1)

        # Display the specific columns
        projected_data_frame = summary_data_frame[['study_id',
                                                   'description',
                                                   'variable_info',
                                                   'number_of_concepts',
                                                   'study_name',
                                                   'permalink']]

        # ??? Might want to dig into this more ...
        top_results = projected_data_frame.head(10)
        study_docs_str = self._format_to_documents_for_llm_context(top_results)
        return study_docs_str

    @staticmethod
    def _format_to_documents_for_llm_context(rows):
        """
        Formats document for llm context
        :param rows:
        :return:
        """
        docs = []
        for _, row in rows.iterrows():
            doc = {
                "page_content": f"{row['description']}\n\n Variable_info: \n{row['variable_info']}",
                "metadata": {
                    "study_id": row['study_id'],
                    "study_name": row['study_name'],
                    "permalink": row['permalink']
                }
            }
            docs.append(doc)
        docs_str = [
            f"\n {doc['metadata']['study_name']} ({doc['metadata']['study_id']}): \n {doc['page_content']}"
            for doc in docs
        ]
        return "\n".join(docs_str)

    def _get_studies_as_runnable(self):
        return RunnableLambda(
            func=lambda x: x,  # fake function, called if we were not async, don't worry we are async here ... :)
            afunc=self._get_study_documents,  # This is the function called in this runnable
            name="retrive_one_hop_variables"
        )

    ######
    #  /// End study retrival
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
    kg_agent = KGChain(config=app_config)
    user_q = Question(chat_history=[], input="What studies are available for asthma?")
    qa_chain = kg_agent.as_generative_chain()
    response = asyncio.run(qa_chain.ainvoke(user_q.dict()))
    print(response)
