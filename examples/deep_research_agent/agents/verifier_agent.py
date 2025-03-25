from pydantic import BaseModel

from agents import Agent

# Agent to sanity‑check a synthesized report for consistency and recall.
# This can be used to flag potential gaps or obvious mistakes.
VERIFIER_PROMPT = (
    "You are a meticulous auditor and editor for the American Economic Review. "
    "You have been contracted out to verify a report that will be handed to a top economist. "
    "You have two jobs. The first is to make sure the report exactly matches the query directives."
    "Here, you should anticipate what the person making the query will respond. Will they be satisfied? "
    "Did they get the information they were after? Infer why they were asking this question and if that need will be met. "
    "Make sure there were no corners cut (e.g., if results are tangential but not related enough). "
    "Your second job is to verify the report is internally consistent, clearly sourced, and makes "
    "no unsupported claims. Point out any issues or uncertainties."
    "You should assume that the person wanting the report cares more about precision than just getting output. "
    " So be very thorough and don't hesitate to push for more if the query isn't met exactly."
)


class VerificationResult(BaseModel):
    verified: bool
    """Whether the report is matching the query exactly and seems coherent and plausible."""

    issues: str
    """If not verified, describe the main issues or concerns. If needed, press for a description of whether the report is obscuring a lack of knowledge by supplmenting with faintly related tangitial facts. """


verifier_agent = Agent(
    name="VerificationAgent",
    instructions=VERIFIER_PROMPT,
    model="gpt-4o",
    output_type=VerificationResult,
)
