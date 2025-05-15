from __future__ import annotations
import asyncio
from typing import List, Literal, NotRequired, TypedDict
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
"""A LangGraph workflow that plans searches, retrieves sources, drafts a report, and
verifies it in a loop until the verifier is satisfied or `max_attempts` is
reached.
"""
# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def extract_text(result) -> str: #Pull plain text out of various LCEL/LLM return shapes.
    if isinstance(result, dict):
        return result.get("text") or result.get("result") or ""
    if hasattr(result, "content"):
        return result.content  # type: ignore[attr-defined]
    return str(result)
# ──────────────────────────────────────────────────────────────────────────────
# Chain: planner → writer → verifier
# ──────────────────────────────────────────────────────────────────────────────
## Planner
planner_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a senior economist. Given the query: '{query}', "
        "list 5‑10 search terms with a brief reason for each to gather recent news, filings, "
        "and analysis."
    ),
)
planner_llm = ChatOpenAI(temperature=0.2, max_tokens=500)
planner_chain = planner_prompt | planner_llm

## Search stub (replace with real tool)

def _search(query: str) -> str:
    return f"SUMMARY: '{query}' (dummy search result)"

search_tool = Tool(
    name="SearchTool",
    func=_search,
    description="Use this to run web searches and retrieve summaries.",
)

## Writer
writer_prompt = PromptTemplate(
    input_variables=["query", "search_results", "feedback_clause"],
    template=(
        "You are a senior economic analyst.\n"
        "QUERY: {query}\n"
        "SEARCH RESULTS:\n{search_results}\n\n"
        "{feedback_clause}\n\n"
        "Compose a concise markdown report with an executive summary and follow‑up research questions. "
        "Cite high‑quality academic or primary sources inline."
    ),
)
writer_llm = ChatOpenAI(temperature=0.2, max_tokens=4096, model_name="gpt-4o")
writer_chain = writer_prompt | writer_llm

## Verifier
verifier_prompt = PromptTemplate(
    input_variables=["query", "report"],
    template=(
        "You are a meticulous copyeditor for the *American Economic Review*.\n"
        "Given the original query: '{query}' and the draft report below,\n"
        "1. Does the report fully satisfy the query?\n"
        "2. Is it internally consistent with no unsupported claims?\n"
        "3. Are citations adequate and frontier‑grade?\n\n"
        "REPORT:\n{report}\n\n"
        "Respond with constructive feedback. Finish with the line 'VERDICT: Satisfied' or 'VERDICT: Not Satisfied'."
    ),
)
verifier_llm = ChatOpenAI(temperature=0.2, max_tokens=1024)
verifier_chain = verifier_prompt | verifier_llm

# ──────────────────────────────────────────────────────────────────────────────
# Graph state
# ──────────────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query: str
    max_attempts: int
    attempt: int
    search_terms: List[str]
    search_results: str
    report: str
    verifier_feedback: str
    status: Literal[
        "planning","searching", "writing", "verifying","complete","failed",
    ]
    feedback_clause: NotRequired[str]

DEFAULT_STATE: ResearchState = {
    "query": "", "max_attempts": 2,  "attempt": 1,  "search_terms": [],  "search_results": "",
    "report": "", "verifier_feedback": "", "feedback_clause": "", "status": "planning",
}

# ──────────────────────────────────────────────────────────────────────────────
# Node implementations
# ──────────────────────────────────────────────────────────────────────────────

def init_state(state: dict) -> ResearchState:
    """Merge user‑supplied state with defaults."""
    merged: ResearchState = {**DEFAULT_STATE, **state}
    return merged

def planning(state: ResearchState) -> ResearchState:
    planner_output = extract_text(planner_chain.invoke({"query": state["query"]}))
    search_terms = [line.strip() for line in planner_output.split("\n") if line.strip()]
    return {**state, "search_terms": search_terms, "status": "searching"}

def searching(state: ResearchState) -> ResearchState:
    results = [_search(term) for term in state["search_terms"]]
    return {**state, "search_results": "\n".join(results), "status": "writing"}

def writing(state: ResearchState) -> ResearchState:
    report = extract_text(
        writer_chain.invoke(
            {
                "query": state["query"], "search_results": state["search_results"],
                "feedback_clause": state.get("feedback_clause", ""),
            }
        )
    )
    return {**state, "report": report, "status": "verifying"}

def verifying(state: ResearchState) -> ResearchState:
    feedback = extract_text(
        verifier_chain.invoke({"query": state["query"], "report": state["report"]})
    )
    satisfied = "VERDICT: Satisfied" in feedback
    exhausted = state["attempt"] >= state["max_attempts"]
    new_status = "complete" if satisfied else ("failed" if exhausted else "writing")

    next_feedback = (
        "Incorporate the verifier feedback below, addressing only unresolved issues:\n" + feedback
        if not satisfied
        else ""
    )
    return {
        **state,
        "verifier_feedback": feedback,
        "feedback_clause": next_feedback,
        "attempt": state["attempt"] + (0 if satisfied else 1),
        "status": new_status,
    }

def _route(state: ResearchState):
    return state["status"]

def _compile() -> StateGraph:
    g = StateGraph(ResearchState)
    g.add_node("init", init_state)
    g.add_node("planning", planning)
    g.add_node("searching", searching)
    g.add_node("writing", writing)
    g.add_node("verifying", verifying)

    g.set_entry_point("init")
    g.add_edge("init", "planning")
    g.add_edge("planning", "searching")
    g.add_edge("searching", "writing")
    g.add_edge("writing", "verifying")
    g.add_conditional_edges(
        "verifying", _route, {"complete": END, "failed": END, "writing": "writing"}
    )
    return g.compile()

# ──────────────────────────────────────────────────────────────────────────────
# Public API required by LangGraph Runtime
# ──────────────────────────────────────────────────────────────────────────────

def run_deep_research(config: RunnableConfig | None = None):
    """Entry point used by LangGraph Runtime (env var `LANGGRAPH_GRAPH_SPEC`)."""
    return _compile()

run_graph = run_deep_research

def _run_cli():
    query = input("Enter your research query: ")
    graph = _compile()
    state_in = {**DEFAULT_STATE, "query": query}
    result = graph.invoke(state_in)
    status = result["status"]
    print("\nFinal status:", status)
    print("\nReport:\n", result["report"])


if __name__ == "__main__":
    _run_cli()
