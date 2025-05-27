import src.config as runtime_config
from langfuse import Langfuse
langfuse = Langfuse(
  secret_key=runtime_config.LANGFUSE_SECRET_KEY,
  public_key=runtime_config.LANGFUSE_PUBLIC_KEY,
  host=runtime_config.LANGFUSE_HOST
)


def get_observations_filtered(observation_filters: dict= None):
    observations = langfuse.fetch_observations(**observation_filters)
    return observations


def get_traces_filtered( observation_name: str,  trace_filters: dict= None):
    # assuming that traces are tagged with code-names for the LLM app...
    # let's first fetch all traces with that application name
    all_traces = langfuse.fetch_traces(**trace_filters)
    # now for each trace ... we will find all the generation observations... this will include
    # data about the input (prompt + context) and the llm answer... reformat them as user_input:... , response: ...

    # flat_responses is our evaluation input... (https://docs.ragas.io/en/stable/getstarted/evals/#evaluating-on-a-dataset)
    flat_responses = []
    for trace in all_traces.data:
        observations = get_observations_filtered({'trace_id': trace.id, 'type': 'GENERATION'})
        if not observations.data:
            continue

        for observation in observations.data:
            if observation.metadata.get('name') == observation_name:
                # append each observation input / output to our dataset...
                flat_responses.append(
                    {
                        # this line might be a little flaky ... is it always true that [0]th element is the one that has
                        # the context and used for generation ...?
                        'user_input': trace.input['input'], #observation.input[0]['content'],
                        'retrieved_contexts': [observation.input[0]['content']],
                        'reference': observation.input[0]['content'],
                        'response': observation.output['content'],
                    }
                )
    # this will be the input for evaluation...
    return flat_responses


# fetch traces:
    # filter the traces by application environment and possibly date range
    # once traces are fetched look for generation observations under each trace...
    # these would give input and output to measure... but how do we know if these observations
    # are part of the answering or not ...
    # one way is changing the koios generation code to add names that are specific to generation steps...
    # other way is to look for parent observations and find generation type observations under those...


if __name__ == '__main__':
    import json
    #TODO parametrize app name, date time range and observation name to filter the datasets for evaluation...
    app_id = "QV_KG_NO_ROUTE"
    observations_name = "answer_generation"
    # these are not used yet ... but need to ...
    filters = {
        'tags': [app_id]
    }

    # watch out for pagenation might need to do something other than the default there...
    obvs = get_traces_filtered(
        observation_name=observations_name,
        trace_filters=filters
    )

    from ragas import EvaluationDataset
    # construct the rages dataset
    evaluation_dataset = EvaluationDataset.from_list(obvs)

    # do evaluation ...
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key= "EMPTY",
        base_url=runtime_config.LLM_URL,
        model=runtime_config.GEN_MODEL_NAME
    )
    evaluator_llm = LangchainLLMWrapper(langchain_llm=llm)
    # evaluation metrics ....
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, AnswerCorrectness


    result = evaluate(dataset=evaluation_dataset, metrics=[
        AnswerCorrectness()
        # Faithfulness(),
        # LLMContextRecall(),
        # FactualCorrectness()
    ],
                      llm=evaluator_llm)

    print(result)
