from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch, RunnablePick, RunnableAssign

from langchain_core.messages import BaseMessage, HumanMessage

from typing import List

from util.llm_helper import LLMFactory



class SupervisorAgent:
    def __init__(self, config):
        self.llm = LLMFactory(config)

        # members of the workflow that are managed by this supervisor
        # routes queries to KG lookup or QV lookup based one extracted intent
        # and user preferences
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
                    "Or should we FINISH? Select one of: {options}",
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
        Ensures response is structured correctly and `next` is a list.
        """
        if not isinstance(response, dict):
            print("Invalid response format. Resetting to default structure.")
            response = {"next": ["KG_lookup"] if query_scope == "single" else ["QV_lookup"]}

        next_agent = response.get("next", [])

        # Ensure `next` is always a list
        if not isinstance(next_agent, list):
            next_agent = [next_agent] if next_agent else ["supervisor"]

        response["next"] = next_agent

        return response



    def as_generative_chain(self):
        from agents.intent_agent_graph import extract_user_preferences_node

        prompt = self._build_prompt()

        def supervisor_logic(state):
            # Short-circuit if already done ---
            if "QV_lookup" in state and state["QV_lookup"].get("input"):
                return {"next": ["FINISH"]}
            if "KG_lookup" in state and state["KG_lookup"].get("input"):
                return {"next": ["FINISH"]}

            # Gather metadata for LLM ---
            scope = state.get("scope", "multiple")
            intents = state.get("intents", [1])
            chat_input = state.get("input", [])
            extra = state.get("extra", {})
            user_prefs = extra.get("user_preferences", {})

            # pass previous output into context if desired
            lookup_results = []
            if "QV_lookup" in state:
                for msg in state["QV_lookup"].get("input", []):
                    lookup_results.append(msg.content)
            if "KG_lookup" in state:
                for msg in state["KG_lookup"].get("input", []):
                    lookup_results.append(msg.content)

            # Run prompt through LLM 
            filled_prompt = prompt.partial(
                scope=scope,
                intents=intents,
                lookup_results="\n".join(lookup_results[-3:]) or "None"  # last few if applicable
            )

            llm_output = self.llm.invoke(filled_prompt.invoke({"input": chat_input}))
            parsed = JsonOutputParser().invoke(llm_output)

            # Enforce and return 
            return self.enforce_user_preferences(parsed, scope)

        return RunnableLambda(supervisor_logic)

