from fastapi import FastAPI
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from langserve import add_routes
from models.user_question import Question
from langchain_core.runnables import RunnableLambda
from guardrails.input_guard import InputGuard
from agents.route_agentic_graph import graph, AgentState


app = FastAPI(
    title="Koios agentic mode"
)

# add langserve routes
add_routes(
    app=app,
    runnable=graph,
    input_type=AgentState,
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)