from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from rich.console import Console

from agents import Runner, RunResult, custom_span, gen_trace_id, trace

from .agents.fronteir_agent import fronteir_agent
from .agents.planner_agent import SearchItem, SearchPlan, planner_agent
from .agents.risk_agent import risk_agent
from .agents.search_agent import search_agent
from .agents.verifier_agent import VerificationResult, verifier_agent
from .agents.writer_agent import ReportData, writer_agent
from .printer import Printer


async def _summary_extractor(run_result: RunResult) -> str:
    """Custom output extractor for sub‑agents that return an AnalysisSummary."""
    # The analyst agents emit an AnalysisSummary with a `summary` field.
    # We want the tool call to return just that summary text so the writer can drop it inline.
    return str(run_result.final_output.summary)


class ResearchManager:
    """
    Orchestrates the full flow: planning, searching, sub‑analysis, writing, and verification,
    with iterative refinement if the verifier flags issues.
    """

    def __init__(self) -> None:
        self.console = Console()
        self.printer = Printer(self.console)

    async def run(self, query: str) -> None:
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            self.printer.update_item(
                "trace_id",
                f"View trace: https://platform.openai.com/traces/{trace_id}",
                is_done=True,
                hide_checkmark=True,
            )
            self.printer.update_item("start", "Starting deep research...", is_done=True)

            verified = False
            iteration = 0
            feedback = ""
            MAX_ITERATIONS = 2

            while not verified and iteration < MAX_ITERATIONS:
                self.printer.update_item("iteration", f"Iteration {iteration + 1}")
                search_plan = await self._plan_searches(query, feedback)
                search_results = await self._perform_searches(search_plan)
                report = await self._write_report(query, search_results, feedback)
                verification = await self._verify_report(query, report)

                verified = verification.verified
                if not verified:
                    feedback = verification.issues
                    self.printer.update_item(
                        f"verifier_feedback_{iteration}",
                        f"Issues detected: {feedback}",
                        is_done=True,
                    )
                iteration += 1

            final_report = f"Report summary\n\n{report.short_summary}"
            self.printer.update_item("final_report", final_report, is_done=True)
            self.printer.end()

        # Print to stdout
        print("\n\n=====REPORT=====\n\n")
        print(f"Report:\n{report.markdown_report}")
        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        print("\n".join(report.follow_up_questions))
        print("\n\n=====VERIFICATION=====\n\n")
        print(verification)

    async def _plan_searches(self, query: str, feedback: str = "") -> SearchPlan:
        self.printer.update_item("planning", "Planning searches...")
        prompt = f"Query: {query}"
        if feedback:
            prompt += f"\nVerifier feedback to address: {feedback}"
        result = await Runner.run(planner_agent, prompt)
        self.printer.update_item(
            "planning",
            f"Will perform {len(result.final_output.searches)} searches",
            is_done=True,
        )
        return result.final_output_as(SearchPlan)

    async def _perform_searches(self, search_plan: SearchPlan) -> Sequence[str]:
        with custom_span("Search the web"):
            self.printer.update_item("searching", "Searching...")
            tasks = [asyncio.create_task(self._search(item)) for item in search_plan.searches]
            results: list[str] = []
            num_completed = 0
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
                self.printer.update_item(
                    "searching", f"Searching... {num_completed}/{len(tasks)} completed"
                )
            self.printer.mark_item_done("searching")
            return results

    async def _search(self, item: SearchItem) -> str | None:
        input_data = f"Search term: {item.query}\nReason: {item.reason}"
        try:
            result = await Runner.run(search_agent, input_data)
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(
        self, query: str, search_results: Sequence[str], feedback: str = ""
    ) -> ReportData:
        fundamentals_tool = fronteir_agent.as_tool(
            tool_name="fundamentals_analysis",
            tool_description="Use to get a short write‑up of key materials",
            custom_output_extractor=_summary_extractor,
        )
        risk_tool = risk_agent.as_tool(
            tool_name="risk_analysis",
            tool_description="Use to get a short write‑up of potential red flags",
            custom_output_extractor=_summary_extractor,
        )
        writer_with_tools = writer_agent.clone(tools=[fundamentals_tool, risk_tool])
        self.printer.update_item("writing", "Thinking about report...")

        input_data = (
            f"Original query: {query}\n"
            f"Verifier feedback to address: {feedback}\n"
            f"Summarized search results: {search_results}"
        )

        result = Runner.run_streamed(writer_with_tools, input_data)
        update_messages = [
            "Planning report structure...",
            "Writing sections...",
            "Finalizing report...",
        ]
        last_update = time.time()
        next_message = 0
        async for _ in result.stream_events():
            if time.time() - last_update > 5 and next_message < len(update_messages):
                self.printer.update_item("writing", update_messages[next_message])
                next_message += 1
                last_update = time.time()
        self.printer.mark_item_done("writing")
        return result.final_output_as(ReportData)

    async def _verify_report(
        self, query: str, report: ReportData
    ) -> VerificationResult:
        self.printer.update_item("verifying", "Verifying report...")
        input_data = f"Original query: {query}\nCurrent Report: {report.markdown_report}"
        result = await Runner.run(verifier_agent, input_data)
        self.printer.mark_item_done("verifying")
        return result.final_output_as(VerificationResult)
