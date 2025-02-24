from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch, RunnablePick, RunnableAssign

from langchain_core.messages import BaseMessage, HumanMessage

from typing import List

from util.llm_helper import LLMFactory

import json


class SupervisorAgent:
    def __init__(self, config):
        self.llm = LLMFactory(config)

        # members of the workflow that are managed by this supervisor
        self.members = {
            "KG_lookup": "This agent identifies biomedical concepts in an input, finds related study variables, "
                         "and provides the study abstracts that include those variables.",
            "QV_lookup": "This agent searches a database of similar questions, "
                         "each linked to potential study abstracts that answer them. "
                         "It then returns study abstracts relevant to the input question."
        }
        self.options = ["FINISH"] + list(self.members.keys())
        # VLLM (our backend llm server) needed a bit of a tweak to send us
        # back a proper json. This config is just to do that.
        self.guided_choice = {
                "type": "object",
                "properties": {
                    "next": {
                        "type": "list",
                        "title": "Next",
                        "anyOf": [
                            {"enum": self.options},
                        ],
                    }
                },
                "required": ["next"],
            }

    def _build_prompt(self):
        # This prompt tells the supervisor what the roles of it's members are so it makes the selection properly.
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers:  {members}.\n {member_description}"
            "Your task is to determine which agent should handle the request next."
            "The user’s intent has been classified as {intents} and their query refers to a {scope} entity/entities."
            "If the request involves a single entity, prefer KG_lookup."
            "If the request involves multiple entities or comparisons, prefer QV_lookup."
            "Return your response as a JSON object with keys 'next' and the value as the choice you made."
        )
        # Our team supervisor is an LLM node. It just picks the next agent to process
        # and decides when the work is completed


        # This is our final prompt. Here we are getting messages either from a User, or other agents through `input`
        # variable and the supervisor will tell the Langraph runtime what (who to call) next.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", system_prompt),
                MessagesPlaceholder(variable_name="input"),
                (
                    "user",
                    "Given the conversation above, which members should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(
            options=str(self.options),
            members=", ".join(self.members.keys()),
            member_description="\n".join([f"{member}: {self.members[member]}" for member in self.members]),
            scope="multiple",  # Defaulting to multiple entities
            intents=[1],   # Default intents to 1 or factual queries 
        )
                
        return prompt


    def enforce_user_preferences(self, response, query_scope):
        """
        Cleans and standardizes the response to enforce scope consistency.
        
        - If the query is about a **single** entity, it forces `KG_lookup`.
        - If the query is about **multiple** entities, it forces `QV_lookup`.
        - Ensures the response is in a clean JSON format.
        """

        # Ensure response is a dictionary --> should be
        if not isinstance(response, dict):
            print("Invalid response format. Resetting to default structure.")
            response = {"next": "KG_lookup" if query_scope == "single" else "QV_lookup"}

        # Extract the next agent decision
        next_agent = response.get("next", "FINISH")

        # Validate based on query scope
        if query_scope == "single" and next_agent == "QV_lookup":
            print(f"Correcting decision: User query was SINGLE entity, but QV_lookup was chosen. Using KG_lookup.")
            response["next"] = "KG_lookup"  # Override choice

        if query_scope == "multiple" and next_agent == "KG_lookup":
            print(f"Correcting decision: User query was MULTIPLE entities, but KG_lookup was chosen. Using QV_lookup.")
            response["next"] = "QV_lookup"  # Override choice

        # Ensure only required fields are in the response
        cleaned_response = {"next": response["next"]}

        return cleaned_response

    def as_generative_chain(self):
        from agents.intent_agent_graph import extract_user_preferences_node
        prompt = self._build_prompt()

        return (
            prompt
            | self.llm  
            | JsonOutputParser()  
            | RunnableLambda(lambda response: self.enforce_user_preferences(response, extract_user_preferences_node({"chat_history": []})  # Default empty chat history

            ))  
        )