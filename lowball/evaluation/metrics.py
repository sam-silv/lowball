"""Evaluation metrics for Lowball."""

from pydantic import BaseModel, Field

from lowball.tasks.schema import TaskInstance


class TaskResult(BaseModel):
    """Result from a single task evaluation."""

    task_id: str
    deal_closed: bool = False
    final_price: float | None = None
    turns_taken: int = 0
    time_elapsed_seconds: float = 0.0
    research_items_found: list[str] = Field(default_factory=list)
    red_flags_detected: list[str] = Field(default_factory=list)
    transcript: list[dict[str, str]] = Field(default_factory=list)

    # Computed scores (filled by compute_metrics)
    deal_score: float | None = None
    budget_adherence: bool | None = None
    negotiation_efficiency: float | None = None
    information_quality: float | None = None
    strategy_score: float | None = None
    red_flag_score: float | None = None
    capitulation_penalty: float | None = None
    overpay_penalty: float | None = None


class BenchmarkResult(BaseModel):
    """Aggregate results across all tasks."""

    agent_id: str
    suite: str
    task_results: list[TaskResult]

    # Aggregates
    completion_rate: float = 0.0
    mean_deal_score: float = 0.0
    mean_negotiation_efficiency: float = 0.0
    mean_information_quality: float = 0.0
    mean_strategy_score: float = 0.0
    budget_adherence_rate: float = 0.0
    mean_red_flag_score: float = 0.0
    mean_composite_score: float = 0.0


def compute_metrics(task: TaskInstance, result: TaskResult) -> TaskResult:
    """Compute all scoring metrics for a completed task."""

    # Deal score: how much below FMV the agent negotiated
    if result.deal_closed and result.final_price is not None:
        result.deal_score = (task.fair_market_value - result.final_price) / task.fair_market_value
    else:
        result.deal_score = -1.0  # Penalty for not closing

    # Budget adherence
    if result.final_price is not None:
        result.budget_adherence = result.final_price <= task.budget_ceiling
    else:
        result.budget_adherence = False

    # Overpay penalty: scaled penalty for exceeding budget (not just boolean)
    if result.final_price is not None and result.final_price > task.budget_ceiling:
        overpay_pct = (result.final_price - task.budget_ceiling) / task.budget_ceiling
        result.overpay_penalty = min(overpay_pct * 5.0, 1.0)  # 20% over budget = max penalty
    else:
        result.overpay_penalty = 0.0

    # Negotiation efficiency: normalized by max turns (1.0 = closed in 1 turn, 0.0 = timed out)
    if result.deal_closed and result.turns_taken > 0:
        result.negotiation_efficiency = max(0.0, 1.0 - (result.turns_taken / task.max_turns))
    else:
        result.negotiation_efficiency = 0.0

    # Capitulation penalty: did the agent accept too quickly without negotiating?
    # Penalize closing in <= 2 turns (likely accepted first counteroffer)
    if result.deal_closed and result.turns_taken <= 2:
        result.capitulation_penalty = 0.5
    elif result.deal_closed and result.turns_taken <= 3:
        result.capitulation_penalty = 0.25
    else:
        result.capitulation_penalty = 0.0

    # Information quality: fraction of required research items discovered
    if task.required_research:
        found = set(r.lower() for r in result.research_items_found)
        required = set(r.lower() for r in task.required_research)
        result.information_quality = len(found & required) / len(required)
    else:
        result.information_quality = 1.0

    # Red flag detection score
    if task.red_flags:
        detected = set(r.lower() for r in result.red_flags_detected)
        flags = set(r.lower() for r in task.red_flags)
        result.red_flag_score = len(detected & flags) / len(flags)
    else:
        result.red_flag_score = 1.0  # No red flags to detect = perfect score

    return result


def composite_score(result: TaskResult) -> float:
    """Compute a single composite score from all metrics.

    Weights deal_score heavily but penalizes for poor research,
    missed red flags, capitulation, and overpaying.
    """
    if result.deal_score is None:
        return -1.0

    score = result.deal_score
    score -= (1.0 - (result.information_quality or 0.0)) * 0.3
    score -= (1.0 - (result.red_flag_score or 0.0)) * 0.3
    score -= result.capitulation_penalty or 0.0
    score -= result.overpay_penalty or 0.0

    return score


def aggregate_results(
    agent_id: str,
    suite: str,
    results: list[TaskResult],
) -> BenchmarkResult:
    """Aggregate individual task results into a benchmark summary."""
    completed = [r for r in results if r.deal_closed]

    return BenchmarkResult(
        agent_id=agent_id,
        suite=suite,
        task_results=results,
        completion_rate=len(completed) / len(results) if results else 0.0,
        mean_deal_score=_mean([r.deal_score for r in results if r.deal_score is not None]),
        mean_negotiation_efficiency=_mean([
            r.negotiation_efficiency for r in results if r.negotiation_efficiency is not None
        ]),
        mean_information_quality=_mean([
            r.information_quality for r in results if r.information_quality is not None
        ]),
        mean_strategy_score=_mean([
            r.strategy_score for r in results if r.strategy_score is not None
        ]),
        budget_adherence_rate=_mean([
            1.0 if r.budget_adherence else 0.0 for r in results if r.budget_adherence is not None
        ]),
        mean_red_flag_score=_mean([
            r.red_flag_score for r in results if r.red_flag_score is not None
        ]),
        mean_composite_score=_mean([composite_score(r) for r in results]),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
