from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch, RunnablePick, RunnableAssign

from langchain_core.messages import BaseMessage, HumanMessage

from typing import List

from util.llm_helper import LLMFactory

import json
from agents.intent_agent_graph import extract_user_preferences_node


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
                    "Given the conversation above, which members  should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(self.options), members=", ".join(self.members.keys()), member_description="\n".join([
            f"{member}: {self.members[member]}" for member in self.members
        ]))
        return prompt

        
    def as_generative_chain(self):
        prompt = self._build_prompt()

        return (
            prompt
            | self.llm  
            | JsonOutputParser()  
            | RunnableLambda(lambda response: self.enforce_user_preferences(
                response, extract_user_preferences_node({"chat_history": self.chat_history})  
            ))  
        ).partial(scope="{scope}")  