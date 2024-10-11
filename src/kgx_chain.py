from fastapi import FastAPI
from typing import List, Tuple
from pydantic import Field, BaseModel
import os
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
from langserve import add_routes

import json
import config
import logging
import requests
import pandas as pd
import redis
from redis.commands.graph import Graph


logging.basicConfig(level=logging.INFO)





app = FastAPI()



class Question(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})

if config.langfuse is None:
    raise ValueError("config.langfuse is not initialized")

template = config.langfuse.get_prompt('ANSWER_GENERATION_PROMPT').prompt

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}")


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# Retrieve studies based on matching concepts
async def retrieve_studies(concepts):
    concepts = concepts.split(',')
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD
    )

    redis_graph = Graph(redis_client, config.REDIS_GRAPH_NAME)

    #Get identifier for all concepts
    def get_concept_identifier(concepts):
        concept_mapping = {}
        for concept in concepts:
            url = "https://sap-qdrant.apps.renci.org/annotate/"
            payload = {
                "text": concept,
                "model_name": "sapbert",
                "count": 1
            }
            response = requests.post(url, json=payload).json()
            if response:
                concept_mapping[concept] = response[0]['curie']
        return concept_mapping



    async def concept_details_kgx(concept_id):
        query = f"""
            MATCH (c {{id: "{concept_id}"}})-[r1]->(v:`biolink.StudyVariable`)-[r2]->(s:`biolink.Study`)
            RETURN c.name AS concept_name,v.name AS variable_name,v.id AS variable_id,v.description AS variable_desc, s.id AS study_id
            LIMIT 100
        """

        result = redis_graph.query(query, read_only=True)
        if result.result_set is None:
            return pd.DataFrame(columns=['concept_name', 'variable_name','variable_id','variable_desc','study_id'])
        data = result.result_set
        df = pd.DataFrame(data, columns=['concept_name', 'variable_name','variable_id','variable_desc','study_id'])
        df['study_id'] = df['study_id'].apply(lambda x: x.split('.')[0])  
        return df


    async def get_study_details(concept_ids):
        dfs = []
        for concept in concept_ids.values():
            df = await concept_details_kgx(concept)
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        if not len(final_df):
            return ""
        df_summary = final_df.groupby('study_id').agg(
            concept_list=('concept_name', lambda x: ', '.join(x)),#.join(sorted(set(x)))),
            variable_desc_list=('variable_desc', lambda x: ', '.join(x)),
            variable_name_list=('variable_name', lambda x: ', '.join(x)),
            variable_id_list=('variable_id', lambda x: ', '.join(x)),
            number_of_concepts=('concept_name', 'nunique')
        ).reset_index()
        df_summary = df_summary.sort_values(by='number_of_concepts', ascending=False)
        df_summary['variable_id_list'] = df_summary['variable_id_list'].apply(
            lambda x: ', '.join(set([vid.split('.')[0] for vid in x.split(', ')]))
        )

      
        def get_study_data(study_id):
            if study_id != None:
                with open(config.STUDIES_JSON_FILE) as stream:  # Replace with the path to your studies JSON file
                    data = json.load(stream)
                for study in data:
                    if study['StudyId'].split('.')[0] == study_id.split('.')[0]:
                        return {
                            "study_name": study['StudyName'],
                            "permalink": study['Permalink'],
                            "description": study['Description'],
                        }

            return {"study_name": "", "permalink": "", "description": ""}

        df_summary[['study_name', 'permalink', 'description']] = df_summary['study_id'].apply(
            lambda x: pd.Series(get_study_data(x)))
        # Filter out rows where study_name is empty (i.e., study not found in the JSON file)
        df_summary = df_summary[df_summary['study_name'] != ""]


        # Concatenate the `variable_name`, `variable_id`, and `variable_desc`
        df_summary['variable_info'] = df_summary.apply(
            lambda row: '\n'.join([f"\t {name} ({var_id}): {desc}"
                                for name, var_id, desc in zip(
                                    row['variable_name_list'].split(', '),
                                    row['variable_id_list'].split(', '),
                                    row['variable_desc_list'].split(', ')
                                )]), axis=1)

        # Display the specific columns
        df_display = df_summary[['study_id', 'description', 'variable_info','number_of_concepts']]
        top_studies = df_display.head(10)

        documents = []

        for _, row in top_studies.iterrows():
            study_data = get_study_data(row['study_id'])
            if study_data:
                # Create the document in dictionary format (JSON serializable)
                doc = {
                    "page_content": f"{row['description']}\n\n Variable_info: \n{row['variable_info']}",
                    "metadata": {
                        "study_id": row['study_id'],
                        "study_name": study_data['study_name'],
                        "permalink": study_data['permalink']
                    }
                }
                documents.append(doc)

        docs_str = [
            f"\n {doc['metadata']['study_name']} ({doc['metadata']['study_id']}): \n {doc['page_content']}"
            for doc in documents
        ]
        return "\n".join(docs_str)
    


    concept_ids = get_concept_identifier(concepts)
    if not concept_ids:
        return "No information available"  # Handle no biomedical concepts case
    study_docs_str = await get_study_details(concept_ids)
    return study_docs_str




# Initialize the chain for concept
def init_concept_chain():
    if config.LLM_SERVER_TYPE == "VLLM":
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_KEY", "EMPTY"),
            base_url=config.LLM_URL,
            model=config.GEN_MODEL_NAME
        )

    elif config.LLM_SERVER_TYPE == "OLLAMA":
        llm = Ollama(
            base_url=config.LLM_URL,
            model=config.GEN_MODEL_NAME
        )
    else:
        raise ValueError(f"Invalid LLM Server type {config.LLM_SERVER_TYPE}")



    get_studies_and_variables = RunnableLambda(func=lambda x: x, afunc=retrieve_studies, name="retrieve_studies_from_kg") 


    # extract concept
    concept_extraction_prompt = config.langfuse.get_prompt("CONCEPT_EXTRACTION_PROMPT").prompt
    ce_prompt = ChatPromptTemplate.from_messages([('system',concept_extraction_prompt)])
    extract_concepts = ce_prompt | llm | StrOutputParser()


    _inputs = RunnableParallel(
        {
            "input": lambda x: x["input"],

            "chat_history": lambda x: _format_chat_history(x['chat_history']),
            "context": extract_concepts | get_studies_and_variables | StrOutputParser()


        }
    )

    answer_generation_chain = RunnableBranch(
        # check if we can get some studies from the graph.
        (
            RunnableLambda(lambda x: print(x) or bool(x.get("context"))).with_config(
                run_name="has_context"
            ),
            ANSWER_PROMPT | llm | StrOutputParser()
        ),
        # If no studies from the graph, and empty context respond with static text
        RunnableLambda(lambda x: "No studies were found to answer the query."),
    )

    qachain = _inputs | answer_generation_chain

    qachain = config.configure_langfuse(qachain)

    return qachain

add_routes(
    app,
    init_concept_chain(),
    path='/dug-kg-chat',
    input_type=Question
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")



