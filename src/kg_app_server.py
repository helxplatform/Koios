from fastapi import FastAPI
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from langserve import add_routes
from models.user_question import Question
from langchain_core.runnables import RunnableLambda
from guardrails.input_guard import InputGuard

input_guard = InputGuard(config=app_config)

# Prep kg app
kg_chain = KGChain(config=app_config)

## Create fastapi app
app = FastAPI(
    title="Koios root server aka Dugbot",
    description="Access kg and question vector chatbots"
)

# add langserve routes
add_routes(
    app=app,
    runnable=input_guard | kg_chain.as_generative_chain(),
    input_type=Question,
    path="/kg-app"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8094)


