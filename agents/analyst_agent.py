from typing import List, Dict, Any
from agents.base_agent import BaseAgent

class AnalystAgent(BaseAgent):
    """
    Agent specialized in technical chart analysis and market patterns.
    Interprets data from the chart recognizer tool.
    """

    def __init__(self):
        super().__init__(
            name="Analyst",
            description="Expert Technical Analyst specialized in price patterns."
        )
        self.system_prompt = (
            "You are the Analyst Agent. You are a professional Technical Analyst AI. "
            "You excel at identifying market reversals and continuations using chart patterns. "
            "When analyzing charts, be precise, mention probability, and provide "
            "actionable technical insights (support, resistance, targets). "
            "Always assume the patterns provided by the vision system are correct."
        )

    async def process(self, messages: List[Dict[str, Any]], tools_enabled: bool = True) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "tool_choice": "none", # Analyst usually doesn't search, it interprets images
        }
