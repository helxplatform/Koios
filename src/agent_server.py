from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.messages import HumanMessage
from agents.route_agentic_lookup_graph import graph
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any, List
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

def format_chat_history(chat_history):
    """
    Formats chat history into conversation pairs.
    
    Args:
        chat_history (list): List of messages alternating between user and dugbot
        
    Returns:
        list: List of paired conversations [user_message, dugbot_response]
    """
    formatted_history = []
    
    try:
        # Process pairs of messages
        for i in range(0, len(chat_history) - 1, 2):
            if i + 1 < len(chat_history):
                formatted_history.append([chat_history[i], chat_history[i+1]])
        
        logger.info(f"Formatted {len(formatted_history)} conversation pairs")
    except Exception as e:
        logger.error(f"Error formatting chat history: {e}")
    
    return formatted_history


def preprocess_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocesses input data for the agent, handling various input formats
    and ensuring chat history is properly formatted.
    
    Args:
        data (Dict[str, Any]): Input data from the client
        
    Returns:
        Dict[str, Any]: Preprocessed data ready for the agent
    """
    try:
        # Handle nested input structure
        if isinstance(data, dict) and "input" in data and isinstance(data["input"], dict):
            inner_input = data["input"]
            
            # Process message objects
            messages = []
            if "input" in inner_input and isinstance(inner_input["input"], list):
                for msg in inner_input["input"]:
                    if isinstance(msg, dict) and msg.get("type") == "human":
                        messages.append(
                            HumanMessage(
                                content=msg.get("content", ""),
                                additional_kwargs=msg.get("additional_kwargs", {}),
                                name=msg.get("name", "user")
                            )
                        )
            
            # Process chat history
            chat_history = []
            if "chat_history" in inner_input and isinstance(inner_input["chat_history"], list):
                try:
                    chat_history = format_chat_history(inner_input["chat_history"])
                except Exception as e:
                    logger.error(f"Chat history formatting error: {e}")
            
            return {
                "input": messages,
                "next": inner_input.get("next", ""),
                "chat_history": chat_history,
                "extra": inner_input.get("extra", {})
            }
        else:
            # Handle direct input format
            if "chat_history" in data and isinstance(data["chat_history"], list):
                try:
                    data["chat_history"] = format_chat_history(data["chat_history"])
                except Exception as e:
                    logger.error(f"Direct chat history formatting error: {e}")
                    data["chat_history"] = []
            
            return data
            
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return {"input": [], "next": "", "chat_history": [], "extra": {}}


# Create a chain with preprocessing input and chathistory
chain = RunnableLambda(preprocess_input) | graph

# Add routes with LangServe
add_routes(
    app=app,
    runnable=chain,
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)