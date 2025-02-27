from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from util.llm_helper import LLMFactory


class SupervisorAgent:
    def __init__(self, config):
        self.llm = LLMFactory(config)

        # members of the workflow that are managed by this supervisor
        self.members = {
            "KG_lookup": "Ideal for queries that are related to study variables (such which variables measures asthma), "
                         "Can answer queries that involve studying relationships between biomedical concepts and related study variables (without needing detailed study descriptions)",
            "QV_lookup": "Best suited for general queries about studies "
                         "Ideal for direct questions about specific study outcomes or findings, where the system can return relevant abstracts based on pre-existing study descriptions "
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
            " following workers:  {members}."
            "\n {member_description}"
            "Your task is to respond the name of workers that should perform the next task."
            "Once the task is completed review it for further action. And respond with the next member to call or FINISH to mark its been done."
            "Return your response as a json object with keys 'next' and the value for that key as the choice you made."
        )
        # Our team supervisor is an LLM node. It just picks the next agent to process
        # and decides when the work is completed


        # This is our final prompt. Here we are getting messages either from a User, or other agents through `input`
        # variable and the supervisor will tell the Langraph runtime what (who to call) next.
        get_user_input = RunnableLambda(lambda x: {"input": [HumanMessage(content=x['input'])]})
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
        return get_user_input | prompt

    def as_generative_chain(self):
        prompt = self._build_prompt()
        return (
                prompt
                | self.llm #.bind(extra_body={"guided_json": self.guided_choice})
                | JsonOutputParser()
        )



