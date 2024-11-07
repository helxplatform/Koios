from fastapi import FastAPI
import config as app_config
from chains.kg_chain import KGChain
from chains.question_lookup_chain import QuestionLookupChain
from langserve import add_routes
from models.user_question import Question
from guardrails.input_guard import InputGuard

input_guard = InputGuard(config=app_config)

# Prep q-vector app
question_vector_chain = QuestionLookupChain(config=app_config)

## Create root app.
app = FastAPI(
    title="Koios root server aka Dugbot",
    description="Access kg and question vector chatbots"
)

# add langserve routes
add_routes(
    app=app,
    runnable=input_guard | question_vector_chain.as_generative_chain(),
    input_type=Question,
    path="/qv-app"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)


