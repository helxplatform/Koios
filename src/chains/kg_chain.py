
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)
import html
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
from util.ner_utils import (
    resolve_to_identifiers_sapbert,
    resolve_to_identifiers_sync
)


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
        self.cypher_query = lambda concept_id, limit: f"""
            MATCH (query_concept {{id: "{concept_id}"}})-[r1]->(variable:`biolink.StudyVariable`)-[r2]->(study:`biolink.Study`)
            RETURN
                 distinct query_concept , r1, variable , r2 , study 
            LIMIT {limit}
            """

    ######
    #  Begin Chain definitions
    ####
    def as_concept_extraction_chain(self):
        return (self.CONCEPT_EXTRACTION_PROMPT | self.llm.with_config(name='concept_extraction') | StrOutputParser()).with_config(
            run_name="concept_extraction")

    def as_retrival_chain(self):
        return RunnableParallel(
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: format_chat_history(x["chat_history"]),
                "context": (self.as_concept_extraction_chain() | self._get_studies_as_runnable() ) #| StrOutputParser())
                .with_config(run_name="retrival")
            }
        )

    def as_generative_chain(self):
        retrival_chain = self.as_retrival_chain().with_config(run_name="kg_lookup_chain")

        answer_chain = RunnableBranch(
            (
                RunnableLambda(
                    lambda x: bool(x)
                ).with_config(
                    run_name="has_context"
                ),
                (self.ANSWER_GENERATION_PROMPT | self.llm | StrOutputParser()).with_config(run_name="answer_generation")
            ),
            # If no studies from the graph, and empty context respond with static text
            RunnableLambda(lambda x: "No studies were found to answer the query.").with_config(run_name="no_data"),
        )

        generative_chain = retrival_chain | RunnableParallel(
            {
                "output": RunnableLambda(lambda x: {
                    "input": x["input"],
                    "context": x.get("context", {}).get("context", ""),
                    "chat_history": x["chat_history"]}
                    ) | answer_chain,
                "extra": RunnableLambda(lambda x: x.get("context", {}).get("extra_data", {}))
            }
        )
        
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

        result = await self.graph.query_graph(self.cypher_query(concept_id, limit))
        return self._format_redis_graph_result(result)

    def _get_one_hop_variables_sync(self, concept_id, limit=100):
        result = self.graph.query_graph_sync(self.cypher_query(concept_id, limit))
        return self._format_redis_graph_result(result)

    def _format_redis_graph_result(self, result):
        if not result.result_set:
            # Return empty table and empty graph.
            return pd.DataFrame(
                columns=['concept_name', 'variable_name', 'variable_id', 'variable_desc', 'study_id']
            ), {"nodes": [], "edges": {}}
        # build data table
        data_table = self.graph.get_results_as_table(result)
        data = [
            [
                data_table['query_concept'][_]['name'],
                data_table['variable'][_]['name'],
                data_table['variable'][_]['id'],
                data_table['variable'][_]['description'],
                data_table['study'][_]['id']

             ] for _ in range(len(data_table['query_concept']))
        ]

        # build knowledge graph
        sub_graph = self.graph.get_results_as_kgx(result)
        # convert table to pandas dataframe
        df = pd.DataFrame(data, columns=['concept_name', 'variable_name', 'variable_id', 'variable_desc', 'study_id'])
        df['study_id'] = df['study_id'].apply(lambda x: x.split('.')[0])
        return df, sub_graph

    def _get_graph_data_sync(self, concept_strings):
        concepts = resolve_to_identifiers_sync(concept_strings.split(','))
        if not concepts:
            return {
                "context": None,
                "extra_data": {}
            }
        results = [self._get_one_hop_variables_sync(concept_curie)
                        for concept_label, concept_curie in concepts.items()]
        return self._get_study_documents(results)

    async def _get_graph_data_async(self, concept_strings):
        concepts = await resolve_to_identifiers_sapbert(concept_strings.split(','))
        if not concepts:
            return {
                "context": None,
                "extra_data": {}
            }
        lookup_tasks = [self._get_one_hop_variables(concept_curie)
                        for concept_label, concept_curie in concepts.items()]

        # results are tuples of a pandas frame , and a graph of edges and nodes
        results = await asyncio.gather(*lookup_tasks)
        return self._get_study_documents(results)

    def _get_study_documents(self, graph_query_results):

        data_frames = [x[0] for x in graph_query_results]

        # pull knowledge graphs for merge
        knowledge_graphs = [x[1] for x in graph_query_results]

        # merge them all
        nodes_processed = set()
        edges_processed = set()
        merged_kg = {"nodes": [], "edges": []}
        for knowledge_graph in knowledge_graphs:
            for node in knowledge_graph["nodes"]:
                if node["id"] not in nodes_processed:
                    merged_kg["nodes"].append(node)
                    nodes_processed.add(node["id"])
            for edge in knowledge_graph["edges"]:
                if edge["id"] not in edges_processed:
                    merged_kg["edges"].append(edge)

        # merge dataframes into a single table
        final_data_frame = pd.concat(data_frames, ignore_index=True)

        if not len(final_data_frame):
            return {
                "context": None,
                "extra_data": {}
            }

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
                               )[0]
            ))

        # filter empty study names,
        summary_data_frame = summary_data_frame[summary_data_frame['study_name'] != ""]

        # Concatenate the `variable_name`, `variable_id`, and `variable_desc`
        summary_data_frame['variable_info'] = summary_data_frame.apply(
            lambda row: '\n'.join([f'\t <variable id="{var_id}">{name} ({var_id}): {desc}</variable>'
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
        return {
            "context": study_docs_str,
            "extra_data": {
                "knowledge_graph": merged_kg
            }
        }

    @staticmethod
    def _format_to_documents_for_llm_context(rows):
        """
        Formats document for llm context
        :param rows:
        :return:
        """
        docs = []
        for _, row in rows.iterrows():
            raw_desc = row['description']
            escaped_desc = html.escape(raw_desc)
            doc = {
                "page_content": f"<abstract>{escaped_desc}</abstract><variables>\n{row['variable_info']}</variables>",
                "metadata": {
                    "study_id": row['study_id'],
                    "study_name": row['study_name'],
                    "permalink": row['permalink']
                }
            }
            docs.append(doc)
        docs_str = [
            (f'<study id="{doc["metadata"]["study_id"]}">'
             f'<title>{doc["metadata"]["study_name"]} ({doc['metadata']["study_id"]}):</title>'
             f'{doc["page_content"]}'
             f'</study>')
            for doc in docs
        ]
        joined_docs = "\n\n".join(docs_str)
        return f"<studies>{joined_docs}</studies>"

    def _get_studies_as_runnable(self):
        return RunnableLambda(
            func=self._get_graph_data_sync,
            afunc=self._get_graph_data_async,
            name="retrieve_one_hop_variables"
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
    # user_q = Question(chat_history=[], input="what studies are there about sickle cell?")
    user_q = Question(chat_history=[], input="variables around choroidal neovascularization")
    qa_chain = kg_agent.as_generative_chain()
    response = asyncio.run(qa_chain.ainvoke(user_q.dict()))
    import json
    print(json.dumps(response, indent=2))
