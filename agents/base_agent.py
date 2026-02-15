from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseAgent(ABC):
    """
    Base class for all specialized agents in the AI Studio.
    Defines the standard interface for reasoning, tool usage, and response generation.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.system_prompt = ""

    @abstractmethod
    async def process(self, messages: List[Dict[str, Any]], tools_enabled: bool = True) -> Dict[str, Any]:
        """
        Process a list of messages and return a response or a plan.
        """
        pass

    def get_system_prompt(self) -> str:
        """
        Return the agent-specific system prompt.
        """
        return self.system_prompt

    def __repr__(self):
        return f"<Agent: {self.name}>"
