from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from agents.route_agentic_lookup_graph import graph, AgentState
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageAdditionalKwargs(BaseModel):
    additionalProp1: dict = Field(default_factory=dict)

class ChatMessage(BaseModel):
    content: str
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    type: str
    name: str
    id: str
    additionalProp1: Dict[str, Any] = Field(default_factory=dict)

class ConfigurableSettings(BaseModel):
    checkpoint_id: str = ""
    checkpoint_ns: str = ""
    thread_id: str = ""

class Config(BaseModel):
    configurable: ConfigurableSettings

class ChatInput(BaseModel):
    input: List[ChatMessage]
    next: str = ""
    chat_history: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    input: ChatInput
    config: Config
    kwargs: Dict[str, Any] = Field(default_factory=dict)

app = FastAPI(title="Koios agentic mode")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_message(message: str) -> str:
    """Clean message by removing graph related content."""
    message = re.sub(r'```[^`]*```', '', message)
    message = re.sub(r'\{[^}]*\}', '', message)
    message = ' '.join(message.split())
    return message

def format_chat_history(messages: List[str]) -> List[List[str]]:
    """Format chat history into pairs of [human_message, ai_message]."""
    formatted_history = []
    for i in range(0, len(messages) - 1, 2):
        human_msg = clean_message(messages[i])
        ai_msg = clean_message(messages[i + 1])
        if human_msg.strip() and ai_msg.strip():
            formatted_history.append([human_msg, ai_msg])
    return formatted_history
@app.post("/agent/invoke")
async def chat_endpoint(request: ChatRequest):
    try:
        messages = [
            HumanMessage(
                content=msg.content,
                additional_kwargs=msg.additional_kwargs,
                name=msg.name,
            ) for msg in request.input.input
        ]

        # Format chat history into pairs
        chat_history = []
        if request.input.chat_history:
            chat_history = format_chat_history(request.input.chat_history)

        logger.info(f"Formatted chat history pairs: {chat_history}")
        input_state = {
            "input": messages,
            "next": request.input.next,
            "chat_history": chat_history,
            "extra": request.input.extra
        }

        thread_config = {
            "configurable": {
                "thread_id": request.config.configurable.thread_id,
                "checkpoint_id": request.config.configurable.checkpoint_id,
                "checkpoint_ns": request.config.configurable.checkpoint_ns
            }
        }

        try:
            result = await graph.ainvoke(
                input_state,
                config=thread_config,
                **request.kwargs
            )
        except Exception as graph_error:
            logger.error(f"Graph processing error: {graph_error}")
            if "string_type" in str(graph_error):
                return {
                    "input": [{
                        "content": request.input.input[0].content if request.input.input else "",
                        "type": "human",
                        "name": "user",
                        "additional_kwargs": {}
                    },
                    {
                        "content": "I'm having trouble processing that question. Could you try rephrasing it?",
                        "type": "ai",
                        "name": "ai",
                        "additional_kwargs": {}
                    }],
                    "extra": {}
                }
            else:
                raise graph_error

        # Format the response
        formatted_messages = []
        
        # First add the original user message
        if request.input.input and len(request.input.input) > 0:
            formatted_messages.append({
                "content": request.input.input[0].content,
                "type": "human",
                "name": "user",
                "additional_kwargs": {}
            })
        
        # Check a valid response
        has_valid_response = False
        ai_content = "I couldn't find any relevant information for that query. Please try asking about a different topic or rephrase your question."
        
        if isinstance(result, dict) and 'input' in result and result['input']:
            messages_list = result['input'] if isinstance(result['input'], list) else [result['input']]

            for msg in reversed(messages_list):
                if hasattr(msg, 'content') and msg.content and msg.content.strip():
                    user_question = request.input.input[0].content if request.input.input else ""
                    if msg.content.strip() != user_question.strip():
                        ai_content = msg.content
                        has_valid_response = True
                        break

        formatted_messages.append({
            "content": ai_content,
            "type": "ai",
            "name": "ai",
            "additional_kwargs": {}
        })

        return {
            "input": formatted_messages,
            "extra": result.get('extra', {}) if isinstance(result, dict) else {}
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)