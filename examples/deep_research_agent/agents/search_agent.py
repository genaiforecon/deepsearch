from agents import Agent, WebSearchTool
from agents.model_settings import ModelSettings

# Given a search term, use web search to pull back a brief summary.
# Summaries should be concise but capture the main financial points.
INSTRUCTIONS = (
    "You are a research assistant specializing in economics, data analysis, and coding. "
    "Given a search term, use web search to retrieve up‑to‑date context and "
    "produce a short summary of at most 300 words."
    "Everything you find must be actionable, self-contained, and relate exactly to the prompt"
    "If you are unable to find anything, state that clearly, and give feedback on what additional info you may need."
)

search_agent = Agent(
    name="FinancialSearchAgent",
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)
