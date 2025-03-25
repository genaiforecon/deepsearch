from pydantic import BaseModel

from agents import Agent

# A sub‑agent specializing in identifying risk factors or concerns.
RISK_PROMPT = (
    "You are an analyst who always tries to go below the surface."
    "Given background research, synthesize and extrapolate the deeper issues and implications of the findings. "
    "Connect to frontier knowledge of the given area. But keep it concise. No more than 3 paragraphs. Focus on higher-level reasoning implications."
)


class AnalysisSummary(BaseModel):
    summary: str
    """Short text summary for this aspect of the analysis."""


risk_agent = Agent(
    name="RiskAnalystAgent",
    instructions=RISK_PROMPT,
    model="o1-preview",
    output_type=AnalysisSummary,
)
