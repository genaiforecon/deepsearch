from pydantic import BaseModel

from agents import Agent

# A sub‑agent focused on analyzing a company's fundamentals.
FINANCIALS_PROMPT = (
    "You are an economics PhD student focused on making sure you are aware of the fronteir of a topic., "
    "Given a collection of web (and optional file) search results, "
    "write a concise analysis of whether you believe these contents constitute the most up to date and high quality sources. "
    "If you are satisfied, simply say that in one sentence. If not, suggest what may be missing and where to find it. Keep it under 2 paragraphs."
)


class AnalysisSummary(BaseModel):
    summary: str
    """Short text summary for this aspect of the analysis."""


financials_agent = Agent(
    name="FundamentalsAnalystAgent",
    instructions=FINANCIALS_PROMPT,
    model="o1-preview",
    output_type=AnalysisSummary,
)
