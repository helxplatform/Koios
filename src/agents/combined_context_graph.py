import functools
from langgraph.graph import END, StateGraph, START
from langfuse.callback import CallbackHandler
from langchain.callbacks.manager import CallbackManager
import config
from agents.utils import *
from models.agent_state import AgentState
from langgraph.checkpoint.memory import MemorySaver
from chains import *
import config as app_config

# Graph structure
"""
                               +-----------+                     
                               | __start__ |                     
                               +-----------+                     
                                      *                          
                                      *                          
                                      *                          
                              +------------+                     
                              | guardrails |.                    
                              +------------+ ....                
                             ..                  ....            
                          ...                        ....        
                        ..                               ....    
      +----------------------------+                         ... 
      | query_relevance_classifier |                           . 
      +----------------------------+                           . 
              ..           ..                                  . 
            ..               ..                                . 
          ..                   ..                              . 
+--------------+                 ..                            . 
| intent_agent |                  .                            . 
+--------------+                  .                            . 
        *                         .                            . 
        *                         .                            . 
        *                         .                            . 
+--------------+            +----------+                     ... 
| lookup_agent |            | doc_node |                 ....    
+--------------+*****       +----------+             ....        
                     *****          *            ....            
                          *****      *       ....                
                               ***   *    ...                    
                                +---------+                      
                                | __end__ |                      
                                +---------+           

"""

qv_kg_lookup_agent_node = functools.partial(agent_node_dict, agent=QVKGChain(app_config).as_generative_chain(),
                                         name="lookup_agent")
intent_agent_node = functools.partial(agent_node_dict, agent=UserIntentChain(app_config).as_generative_chain(),
                                      name="intent_agent")
query_routing_node = functools.partial(agent_node_dict, agent=QueryRelevanceClassifierChain(app_config).as_generative_chain(),
                                      name="query_relevance_classifier")
doc_node = functools.partial(agent_node_dict, agent=DocumentationGenerationChain(app_config).as_generative_chain(),
                                      name="doc_node")
# Initialize the workflow with our state schema.
workflow = StateGraph(AgentState)


# workflow.add_node("guardrails", guardrails_node)
workflow.add_node("lookup_agent", qv_kg_lookup_agent_node)
workflow.add_node("intent_agent", intent_agent_node)
workflow.add_node("query_relevance_classifier", query_routing_node)
workflow.add_node("doc_node", doc_node)

workflow.add_edge(START, "query_relevance_classifier")
# workflow.add_edge("guardrails", "query_relevance_classifier")
# workflow.add_conditional_edges("guardrails", lambda x: x.next,
                            #    {"continue": "query_relevance_classifier", "FINISH": END})
workflow.add_conditional_edges("query_relevance_classifier", lambda x: x.next,
                               {"lookup": "intent_agent", "documentation": "doc_node"})
workflow.add_edge("intent_agent", "lookup_agent")
workflow.add_edge("lookup_agent", END)
workflow.add_edge("doc_node", END)

langfuse_callback = CallbackHandler(
    host=config.LANGFUSE_HOST,
    secret_key=config.LANGFUSE_SECRET_KEY,
    public_key=config.LANGFUSE_PUBLIC_KEY
)
callback_manager = CallbackManager([langfuse_callback])
# Compile the graph with memory
graph = workflow.compile(checkpointer=MemorySaver()) #.with_config(callbacks=callback_manager)


# ---- test code ----
async def main():
    graph.get_graph().print_ascii()
    from langfuse.callback import CallbackHandler
    langfuse_callback = CallbackHandler(
        host=config.LANGFUSE_HOST,
        secret_key=config.LANGFUSE_SECRET_KEY,
        public_key=config.LANGFUSE_PUBLIC_KEY
    )
    thread_config = {"configurable": {"thread_id": "1"}, "callbacks": [langfuse_callback]}
    result = await graph.ainvoke(
            {
                # Mimicking previous interactions.
                "chat_history": [
                    # ("wHr ", "the heart is melting"),
                ],
                # Current question.
                "input": "What kind of question should i ask?",

            }, config=thread_config
    )


if __name__ == "__main__":
    # Test code to run
    import asyncio
    asyncio.run(main())
# --- end test ---