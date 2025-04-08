from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

@dataclass
class WebSearchTool:
    name: str = "web_search"
    description: str = "Search the web for up-to-date information."
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

@dataclass
class FunctionToolResult:
    name: str
    arguments: Dict[str, Any]
    result: Any

@dataclass
class Tool:
    name: str
    description: str
    function: Optional[Callable] = None
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input for the tool"
                        }
                    },
                    "required": ["input"]
                }
            }
        }

def function_tool(name_override: str = None, description_override: str = ""):
    def decorator(func):
        return Tool(
            name=name_override or func.__name__,
            description=description_override or func.__doc__ or "",
            function=func
        )
    return decorator