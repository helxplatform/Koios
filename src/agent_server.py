from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import langfuse
import config
from agents.combined_context_graph import graph
import logging

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