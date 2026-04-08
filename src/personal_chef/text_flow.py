from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from personal_chef.tools import web_search
from langchain_core.messages import HumanMessage
from pathlib import Path
from uuid import uuid4

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent

gpt_5_nano = init_chat_model(model="gpt-5-nano")

with open(PROJECT_ROOT / "docs" / "system-prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

chef_agent = create_agent(
    model=gpt_5_nano,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    tools=[web_search]
)

ingredients = "broccoli, red peppers, chicken thighs, lemon, brown rice"
user_prompt = HumanMessage([
    {"type": "text", "text": f"The following are leftover ingredients currently present in my fridge: {ingredients}"}
])

thread_id = str(uuid4())
mem_config = {"configurable": {"thread_id": thread_id}}

for token, metadata in chef_agent.stream(
    {"messages": [user_prompt]},
    config=mem_config,
    stream_mode="messages"
):
    if token.content:
        print(token.content, end="", flush=True)

followup_question = HumanMessage("Do you think I could add a sauce to the first recipe? What would you suggest?")
followup_answer = chef_agent.invoke({"messages": [followup_question]}, config=mem_config)

state = chef_agent.get_state(mem_config)
messages = state.values["messages"]

output_dir = PROJECT_ROOT / "conversations"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"conversation_{thread_id}.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for msg in messages:
        f.write(f"[{msg.type.upper()}]\n{msg.content}\n\n")
