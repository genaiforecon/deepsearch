from dataclasses import dataclass

@dataclass
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = False
    truncation: str | None = None
    max_tokens: int | None = None
