from tools.web_browse import (
    web_browse, WEB_BROWSE_TOOL_DEFINITION,
    web_search, WEB_SEARCH_TOOL_DEFINITION,
)
from tools.deep_research import deep_research, DEEP_RESEARCH_TOOL_DEFINITION

TOOL_REGISTRY = {
    "web_search": {
        "handler": web_search,
        "definition": WEB_SEARCH_TOOL_DEFINITION,
    },
    "web_browse": {
        "definition": WEB_BROWSE_TOOL_DEFINITION,
        "handler": web_browse,
    },
    "deep_research": {
        "definition": DEEP_RESEARCH_TOOL_DEFINITION,
        "handler": deep_research,
    },
}


def get_tool_definitions() -> list[dict]:
    """Return OpenAI-compatible tool definitions for all registered tools."""
    return [entry["definition"] for entry in TOOL_REGISTRY.values()]


async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a registered tool by name and return its string result."""
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return f"Error: Unknown tool '{name}'"
    try:
        return await entry["handler"](**arguments)
    except Exception as e:
        return f"Error executing tool '{name}': {e}"
