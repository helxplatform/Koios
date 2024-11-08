from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from util.llm_helper import LLMFactory


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
            " following workers:  {members}."
            "\n {member_description}"
            "Your task is to respond the name of workers that should perform the task next."
            "Once the task is completed review it for further action. And respond with the next member to call or FINISH to mark its been done."
        )
        # Our team supervisor is an LLM node. It just picks the next agent to process
        # and decides when the work is completed


        # This is our final prompt. Here we are getting messages either from a User, or other agents through `input`
        # variable and the supervisor will tell the Langraph runtime what (who to call) next.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="input"),
                (
                    "system",
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
                | self.llm.bind(extra_body={"guided_json": self.guided_choice})
                | JsonOutputParser()
        )



