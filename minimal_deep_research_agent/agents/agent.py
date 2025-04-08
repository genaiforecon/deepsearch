from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type, TypeVar, Generic, Protocol, Union
from .model_settings import ModelSettings
from .tool import Tool

T = TypeVar('T')

class Context(Protocol):
    pass

@dataclass
class Agent(Generic[T]):
    name: str
    instructions: str
    model: Optional[str] = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: List[Tool] = field(default_factory=list)
    output_type: Optional[Type] = None
    handoffs: List[Any] = field(default_factory=list)
    input_guardrails: List[Any] = field(default_factory=list)
    output_guardrails: List[Any] = field(default_factory=list)
    hooks: Any = None
    
    def clone(self, **kwargs) -> 'Agent[T]':
        """Create a copy of this agent with updated parameters"""
        new_agent = Agent(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            model_settings=self.model_settings,
            tools=self.tools.copy(),
            output_type=self.output_type,
            handoffs=self.handoffs.copy(),
            input_guardrails=self.input_guardrails.copy(),
            output_guardrails=self.output_guardrails.copy(),
            hooks=self.hooks
        )
        for key, value in kwargs.items():
            setattr(new_agent, key, value)
        return new_agent
    
    def as_tool(self, tool_name: str, tool_description: str, custom_output_extractor: Optional[Callable] = None) -> Tool:
        """Convert this agent to a tool that can be used by other agents"""
        return Tool(
            name=tool_name,
            description=tool_description,
            function=None  # In a full implementation this would be a function
        )
    
    async def get_system_prompt(self, context: Any) -> str:
        """Get the system prompt for this agent"""
        return self.instructions