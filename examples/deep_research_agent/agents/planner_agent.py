from pydantic import BaseModel

from agents import Agent

PROMPT = (
    "You are a senior economist at the world bank. Given a research request, "
    "produce a set of web searches to gather the context needed. Aim for recent "
    "research, tweets, blog posts, from the frontier of knowledge. "
    "Output between 5 and 15 search terms to query for."
    " The goal is that these terms should span the query and provide high-quality content."
)


class SearchItem(BaseModel):
    reason: str
    """Your reasoning for why this search is relevant."""

    query: str
    """The search term to feed into a web (or file) search."""


class SearchPlan(BaseModel):
    searches: list[SearchItem]
    """A list of searches to perform."""


planner_agent = Agent(
    name="PlannerAgent",
    instructions=PROMPT,
    model="o3-mini",
    output_type=SearchPlan,
)
