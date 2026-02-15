from tools.web_browse import (
    web_browse, WEB_BROWSE_TOOL_DEFINITION,
    web_search, WEB_SEARCH_TOOL_DEFINITION,
)

TOOL_REGISTRY = {
    "web_search": {
        "handler": web_search,
        "definition": WEB_SEARCH_TOOL_DEFINITION,
    },
    "web_browse": {
        "handler": web_browse,
        "definition": WEB_BROWSE_TOOL_DEFINITION,
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
        handler_coro = entry["handler"](**arguments)
        # Check if it's an async generator
        if hasattr(handler_coro, "__aiter__"):
            final_result = ""
            async for event in handler_coro:
                if event["type"] == "result":
                    final_result = event["content"]
            return final_result
        return await handler_coro
    except Exception as e:
        return f"Error executing tool '{name}': {e}"
