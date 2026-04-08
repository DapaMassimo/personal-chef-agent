from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
tavily_client = TavilyClient()

@tool("web_search", description="Search the web for information")
def web_search(query: str) -> Dict[str, Any]:
    return tavily_client.search(query)
