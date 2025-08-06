from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import langfuse
import config
from agents.combined_context_graph import graph
import logging
from models.user_question import SimpleQuery
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Create FastAPI app
app = FastAPI(
    title="Koios agentic mode"
)

# Add CORS middleware for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add routes with LangServe
add_routes(
    app=app,
    runnable=graph,
    path=config.SERVER_ROOT_URL.rstrip("/")
)

@app.post(f"{config.SERVER_ROOT_URL.rstrip('/')}/invoke_test")
async def invoke_full(query: SimpleQuery):
    thread_config = {"configurable": {"thread_id": "1"},}

    response = await graph.ainvoke(
            {
                # Mimicking previous interactions.
                "chat_history": [],
                # Current question.
                "input": query.query,
                "return_prompt": True

            }, config=thread_config)
    formatted_response = {
        "user_input": query.query,
        "response": response["output"].content,
        "retrieved_contexts": [x.content for x in response["extra"]["context"].messages]
    }
    return formatted_response


@app.get(f"{config.SERVER_ROOT_URL.rstrip('/')}/score/{{trace_id}}/{{score}}")
async def trace(trace_id: str, score: str):
    """
    Give a score to a trace
    :param trace_id:
    :param score:
    :return:
    """

    langfuse_client = langfuse.Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
        environment=config.ENVIRONMENT
    )
    langfuse_client.score(
        name="user_feedback",
        id=f"{trace_id}",
        trace_id=trace_id,
        value=score,
        data_type="CATEGORICAL",
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)