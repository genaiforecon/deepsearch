import os
import getpass

from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain_openai import ChatOpenAI  # Updated import

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Helper: Extract text from result objects.
def extract_text(result):
    if isinstance(result, dict):
        return result.get("text") or result.get("result") or ""
    elif hasattr(result, "content"):
        return result.content
    return str(result)

# 1. Define the planner chain using the pipe operator.
planner_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a senior economist. Given the query: '{query}', "
        "list 5 to 10 search terms with a brief reason for each to gather recent news, filings, "
        "and analysis."
    )
)
planner_llm = ChatOpenAI(temperature=0.2, max_tokens=500)
planner_chain = planner_prompt | planner_llm  # RunnableSequence

# 2. Define a search tool.
def search_tool_func(query: str) -> str:
    # Placeholder integration for a web search API.
    return f"Summary for search query: {query}"

search_tool = Tool(
    name="SearchTool",
    func=search_tool_func,
    description="Use this tool to perform web searches for relevant sources, data, and info."
)

# 3. Define the writer chain using the pipe operator.
writer_prompt = PromptTemplate(
    input_variables=["query", "search_results", "feedback_clause"],
    template=(
        "You are a senior economic analyst. Given the query: '{query}', "
        "the following search summaries:\n{search_results}\n"
        "{feedback_clause} "
        "Produce a markdown report with an executive summary and follow‑up research questions."
        "Be sure to include citations. Place high emphasis on high-quality academic citations when appropriate."
    )
)
writer_llm = ChatOpenAI(temperature=0.2, max_tokens=15000, model_name='gpt-4o')
writer_chain = writer_prompt | writer_llm  # RunnableSequence

# 4. Define the verifier chain using the pipe operator.
verifier_prompt = PromptTemplate(
    input_variables=["query", "report"],
    template=(
        "You are a meticulous auditor and editor for the American Economic Review. "
        "You have been contracted to verify a report that will be handed to a top economist. "
        "Your first job is to ensure the report exactly matches the query directives. "
        "Anticipate what the person making the query will expect. Did they get the information they needed? "
        "Make sure no corners were cut (for example, tangential results that aren’t closely related). "
        "Your second job is to verify the report is internally consistent, clearly sourced, and makes no unsupported claims. "
        "Point out any issues or uncertainties. Make sure sources cited are from the fronteir of knowledge (and/or academic economics, when appropriate)."
        "Given the original query: '{query}' and the generated report: '{report}', provide your detailed assessment. "
        "At the end of your assessment, include a line 'VERDICT: Satisfied' if the report fully meets the query, "
        "or 'VERDICT: Not Satisfied' otherwise."
    )
)
verifier_llm = ChatOpenAI(temperature=0.2, max_tokens=1000)
verifier_chain = verifier_prompt | verifier_llm  # RunnableSequence

# 5. Orchestrate the workflow with a verification loop.
def run_deep_research(query: str, max_attempts: int = 2):
    # Step 1: Generate search queries.
    planner_result = planner_chain.invoke({"query": query})
    planner_output = extract_text(planner_result)
    search_terms = [term.strip() for term in planner_output.split('\n') if term.strip()]

    # Step 2: Run searches and collect summaries.
    search_results_list = []
    for term in search_terms:
        result = search_tool.func(term)
        search_results_list.append(result)
    search_results = "\n".join(search_results_list)

    feedback_clause = ""  # Initially, no verifier feedback.
    attempt = 1

    while attempt <= max_attempts:
        print(f"Attempt {attempt}:")

        # Step 3: Writer produces the report.
        writer_result = writer_chain.invoke({
            "query": query,
            "search_results": search_results,
            "feedback_clause": feedback_clause
        })
        report = extract_text(writer_result)

        # Step 4: Verifier reviews the report.
        verifier_result = verifier_chain.invoke({
            "query": query,
            "report": report
        })
        verifier_feedback = extract_text(verifier_result)
        print("Verifier Feedback:\n", verifier_feedback)

        # Check for verdict in the verifier's output.
        if "VERDICT: Satisfied" in verifier_feedback:
            print("Final report accepted by verifier.\n")
            return report
        else:
            # Pass verifier feedback back to writer in next iteration.
            feedback_clause = f"Incorporate the following verifier feedback into your revised report:\n{verifier_feedback}"
            print("Revising report based on verifier feedback...\n")
            attempt += 1

    # If max attempts reached, return the last report with a warning.
    print("Maximum attempts reached. Returning the last generated report.")
    return report

if __name__ == "__main__":
    query = input("Enter your research query: ")
    final_report = run_deep_research(query)
    print("\nFinal Report:\n", final_report)

