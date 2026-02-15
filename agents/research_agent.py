from typing import List, Dict, Any
from agents.base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
    """
    Agent specialized in web search and factual information retrieval.
    Prefers using the web_search tool to verify facts before answering.
    """

    def __init__(self):
        super().__init__(
            name="Researcher",
            description="Expert at web searches and factual verification."
        )
        self.system_prompt = (
            "You are the Researcher Agent. Expert at web searches and factual verification. "
            "Your primary goal is to provide accurate, up-to-date information. "
            "Use the 'web_search' tool for news or factual data. "
            "IMPORTANT: If a search tool returns a 'Synthesis', 'Summary', or sufficient information, "
            "DO NOT search again for the same query. Answer the user's question directly based on the tool output."
        )

    async def process(self, messages: List[Dict[str, Any]], tools_enabled: bool = True) -> Dict[str, Any]:
        # For now, the routing logic remains in chat.py, 
        # but the agent provides the strategy and system prompt.
        return {
            "system_prompt": self.system_prompt,
            "tool_choice": "auto" if tools_enabled else "none",
        }
