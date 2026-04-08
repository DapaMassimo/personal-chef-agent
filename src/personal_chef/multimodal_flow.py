from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from personal_chef.tools import web_search
from langchain_core.messages import HumanMessage
from pathlib import Path
from uuid import uuid4
import base64
from pprint import pprint

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent

gpt_5_nano = init_chat_model(model="gpt-5-nano")

with open(PROJECT_ROOT / "docs" / "multimodal-system-prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

chef_agent = create_agent(
    model=gpt_5_nano,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    tools=[web_search]
)

file_path = PROJECT_ROOT / "images" / "fridge.png"

with open(file_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Analyze this image of my fridge contents and provide me with some recipes using only those ingredients."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
])

thread_id = str(uuid4())
multimodal_config = {"configurable": {"thread_id": thread_id}}

multimodal_response = chef_agent.invoke(
    {"messages": [multimodal_question]},
    config=multimodal_config
)

pprint(multimodal_response)
print("\n\n\n")
print(multimodal_response['messages'][-1].content)

followup_question = HumanMessage("Do you think I could add a sauce to the first recipe? What would you suggest?")
followup_answer = chef_agent.invoke({"messages": [followup_question]}, config=multimodal_config)

state = chef_agent.get_state(multimodal_config)
messages = state.values["messages"]

output_dir = PROJECT_ROOT / "conversations"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"conversation_multimodal_{thread_id}.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for msg in messages:
        f.write(f"[{msg.type.upper()}]\n{msg.content}\n\n")
