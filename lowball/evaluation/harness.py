"""Evaluation harness — orchestrates task execution and scoring."""

import time
from typing import Any, Protocol

from lowball.environment.messaging import MessagingEnvironment
from lowball.environment.tools import MarketplaceToolDispatcher
from lowball.evaluation.judge import judge_negotiation, judge_research
from lowball.evaluation.metrics import (
    BenchmarkResult,
    TaskResult,
    aggregate_results,
    compute_metrics,
)
from lowball.simulation.market import Marketplace
from lowball.simulation.seller import SimulatedSeller
from lowball.tasks.schema import TaskInstance


class AgentInterface(Protocol):
    """Protocol that agents under evaluation must implement."""

    async def setup(self, task_description: str, tools: MarketplaceToolDispatcher) -> None:
        """Called once at the start of a task with the goal and marketplace tools."""
        ...

    async def browse_and_select(self) -> str | None:
        """Browse the marketplace via tools and return a listing_id to negotiate on, or None."""
        ...

    async def negotiate(self, messaging: MessagingEnvironment, max_turns: int) -> None:
        """Run the full negotiation loop using tool calls against the messaging environment."""
        ...

    async def report_research(self) -> list[str]:
        """Return list of research findings the agent discovered."""
        ...

    async def report_red_flags(self) -> list[str]:
        """Return list of red flags/suspicious listings the agent identified."""
        ...


class EvaluationHarness:
    """Runs Lowball tasks against an agent and produces scored results."""

    def __init__(
        self,
        agent: AgentInterface,
        llm_client: Any = None,
    ) -> None:
        self.agent = agent
        self.llm_client = llm_client

    async def run_task(self, task: TaskInstance) -> TaskResult:
        """Execute a single task and return scored results."""
        start_time = time.monotonic()

        # Set up marketplace tools
        marketplace = Marketplace()
        tools = MarketplaceToolDispatcher(marketplace)

        # Phase 1: Agent browses marketplace via tools
        await self.agent.setup(task.description, tools)
        selected_listing = await self.agent.browse_and_select()

        if selected_listing is None:
            return self._empty_result(task, time.monotonic() - start_time)

        # Phase 2: Negotiation via tool calls
        seller = SimulatedSeller(task.seller, llm_client=self.llm_client)
        messaging = MessagingEnvironment(seller)

        await self.agent.negotiate(messaging, max_turns=task.max_turns)

        # Phase 3: Collect results
        research = await self.agent.report_research()
        red_flags = await self.agent.report_red_flags()

        result = TaskResult(
            task_id=task.task_id,
            deal_closed=messaging.state.deal_closed,
            final_price=messaging.deal_price,
            turns_taken=len(messaging.history),
            time_elapsed_seconds=time.monotonic() - start_time,
            research_items_found=research,
            red_flags_detected=red_flags,
            transcript=messaging.get_transcript(),
        )

        # Score
        result = compute_metrics(task, result)

        # LLM judge for strategy scoring
        if self.llm_client and result.transcript:
            judge_result = await judge_negotiation(
                result.transcript,
                context=f"Vehicle: {task.vehicle.make} {task.vehicle.model}, "
                f"FMV: ${task.fair_market_value:,.0f}, "
                f"Budget: ${task.budget_ceiling:,.0f}",
                llm_client=self.llm_client,
            )
            result.strategy_score = float(judge_result.get("overall_score", 0)) / 10.0

        # LLM judge for research and red flag evaluation
        if self.llm_client and (task.required_research or task.red_flags):
            research_result = await judge_research(
                research_findings=research,
                red_flags_detected=red_flags,
                required_research=task.required_research or [],
                known_red_flags=task.red_flags or [],
                llm_client=self.llm_client,
            )
            result.information_quality = float(research_result.get("research_score", 0))
            result.red_flag_score = float(research_result.get("red_flag_score", 0))

        return result

    async def run_suite(self, tasks: list[TaskInstance], agent_id: str, suite: str) -> BenchmarkResult:
        """Run all tasks in a suite sequentially and return aggregate results."""
        results: list[TaskResult] = []
        for task in tasks:
            result = await self.run_task(task)
            results.append(result)
        return aggregate_results(agent_id, suite, results)

    def _empty_result(self, task: TaskInstance, elapsed: float) -> TaskResult:
        result = TaskResult(
            task_id=task.task_id,
            deal_closed=False,
            time_elapsed_seconds=elapsed,
        )
        return compute_metrics(task, result)
