from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import config


# members of the workflow that are managed by this supervisor
members = ["comedian", "researcher"]

# This prompt tells the supervisor what the roles of it's members are so it makes the selection properly.
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}."
    "The researcher is responsible for looking up studies and giving you answers and is not good for telling jokes."
    "The comedian is responsible for telling you jokes but is not good for telling you facts."
    "Your task is to respond the name of workers that should perform the task next."
    "Once the task is completed review it for further action. And respond with the next member to call or FINISH to mark its been done."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

# VLLM (our backend llm server) needed a bit of a tweak to send us back a proper json. This config is just to do that.
guided_choice = {
        "type": "object",
        "properties": {
            "next": {
                "type": "list",
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    }

# This is our final prompt. Here we are getting messages either from a User, or other agents through `input`
# variable and the supervisor will tell the Langraph runtime what (who to call) next.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="input"),
        (
            "system",
            "Given the conversation above, which memebers  should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# basic llm defination (pointing to vllm llama3.1-8b)
llm = ChatOpenAI(
            api_key="EMPTY",
            base_url=config.LLM_URL,
            model=config.GEN_MODEL_NAME,
            )

# Put this thing together as a chain.
supervisor_chain = (
    prompt
    | llm.bind(extra_body={"guided_json": guided_choice})
    | JsonOutputParser()
)
