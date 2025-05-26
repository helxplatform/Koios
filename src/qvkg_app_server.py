from fastapi import FastAPI
import config as app_config
from langserve import add_routes
from models.user_question import Question
from guardrails.input_guard import InputGuard
from chains.qvkg_chain import QVKGChain


# Create a FastAPI app with the combined chain
input_guard = InputGuard(config=app_config)
qvkg_chain = QVKGChain(config=app_config)

app = FastAPI(
    title="Koios Combined qvkg Server",
    description="Access combined knowledge graph and question vector chatbot"
)

# Add langserve routes
add_routes(
    app=app,
    runnable=input_guard | qvkg_chain.as_generative_chain(),
    input_type=Question,
    path="/qvkg-app"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)


