# Personal Chef

**FridgeChef** — an AI agent that turns leftover fridge ingredients into recipes. Give it a list of ingredients (or a photo of your fridge) and it searches the web for ideas, adapts them to what you actually have, and replies with full recipes, citations, and tips.

Built with [LangChain](https://www.langchain.com/) / LangGraph agents, OpenAI's `gpt-5-nano`, and [Tavily](https://tavily.com/) for web search.

## Features

- **Text flow** ([src/personal_chef/text_flow.py](src/personal_chef/text_flow.py)) — pass a comma-separated ingredient list as a `HumanMessage`, stream the agent's response token-by-token, then ask follow-up questions on the same thread.
- **Multimodal flow** ([src/personal_chef/multimodal_flow.py](src/personal_chef/multimodal_flow.py)) — send a base64-encoded fridge photo (`images/fridge.png`) and let the model identify ingredients before suggesting recipes.
- **Conversation memory** — both flows use LangGraph's `InMemorySaver` checkpointer keyed by a `thread_id`, so follow-up turns retain context. Full transcripts are written to [conversations/](conversations/).
- **Web search tool** ([src/personal_chef/tools.py](src/personal_chef/tools.py)) — a thin `@tool` wrapper around the Tavily client, exposed to the agent as `web_search`.
- **System prompts** — the chef's persona, workflow, and response format live in [docs/system-prompt.txt](docs/system-prompt.txt) and [docs/multimodal-system-prompt.txt](docs/multimodal-system-prompt.txt).

## Project layout

```

|__src/personal_chef/
│   ├── text_flow.py         # text-only ingredient → recipes flow
│   ├── multimodal_flow.py   # image-based fridge → recipes flow
│   └── tools.py             # Tavily web_search tool
├── docs/                    # system prompts
├── images/                  # sample fridge photos
├── conversations/           # saved transcripts (per thread_id)
└── pyproject.toml
```

## Requirements

- Python ≥ 3.12
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- A `.env` file at the project root (loaded via `python-dotenv` in both flows and in [tools.py](src/personal_chef/tools.py)):

  ```dotenv
  # Required
  OPENAI_API_KEY='sk-...'
  TAVILY_API_KEY='tvly-...'

  # Optional — LangSmith tracing/evaluation
  LANGSMITH_API_KEY='lsv2_...'
  LANGSMITH_TRACING=true
  LANGSMITH_PROJECT=leftover_personal_chef
  ```

  The `LANGSMITH_*` vars are optional: set them to send traces of each agent run to your [LangSmith](https://smith.langchain.com/) project for debugging and evaluation. Leave them commented out to run fully offline of LangSmith.

Dependencies (see [pyproject.toml](pyproject.toml)): `langchain`, `langchain-openai`, `langchain-tavily`, `tavily-python`, `python-dotenv`.

## Setup

```bash
uv sync
```

## Usage

Run the text flow:

```bash
uv run python -m personal_chef.text_flow
```

Run the multimodal flow (expects `images/fridge.png`):

```bash
uv run python -m personal_chef.multimodal_flow
```

Each run creates a new `thread_id` and saves the full conversation to `conversations/conversation_<thread_id>.txt` (or `conversation_multimodal_<thread_id>.txt`).

To try your own ingredients, edit the `ingredients` string in [text_flow.py](src/personal_chef/text_flow.py), or drop a different image into `images/` and update the path in [multimodal_flow.py](src/personal_chef/multimodal_flow.py).
