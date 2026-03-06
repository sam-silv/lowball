"""Tests for evaluation metrics."""

from lowball.evaluation.metrics import TaskResult, compute_metrics, aggregate_results, composite_score
from lowball.tasks.loader import load_task


def test_deal_score_below_fmv() -> None:
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=17000,
        turns_taken=8,
        research_items_found=["fair market value", "vehicle history check", "recall status", "mileage verification"],
    )
    scored = compute_metrics(task, result)
    assert scored.deal_score is not None
    assert scored.deal_score > 0  # Below FMV
    assert scored.budget_adherence is True
    assert scored.information_quality == 1.0


def test_deal_score_above_fmv() -> None:
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=19000,
        turns_taken=5,
    )
    scored = compute_metrics(task, result)
    assert scored.deal_score is not None
    assert scored.deal_score < 0  # Above FMV
    assert scored.budget_adherence is True  # Still within budget (19000 <= 19000)


def test_no_deal_penalty() -> None:
    task = load_task("easy/civic_2019")
    result = TaskResult(task_id=task.task_id, deal_closed=False)
    scored = compute_metrics(task, result)
    assert scored.deal_score == -1.0
    assert scored.budget_adherence is False
    assert scored.negotiation_efficiency == 0.0


def test_budget_exceeded() -> None:
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=20000,  # Over 19k budget
        turns_taken=3,
    )
    scored = compute_metrics(task, result)
    assert scored.budget_adherence is False
    assert scored.overpay_penalty is not None
    assert scored.overpay_penalty > 0


def test_capitulation_penalty_quick_close() -> None:
    """Agent that closes in 2 turns or fewer should be penalized."""
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=18000,
        turns_taken=2,
    )
    scored = compute_metrics(task, result)
    assert scored.capitulation_penalty == 0.5


def test_no_capitulation_penalty_normal_negotiation() -> None:
    """Agent that takes 4+ turns should not be penalized."""
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=18000,
        turns_taken=6,
    )
    scored = compute_metrics(task, result)
    assert scored.capitulation_penalty == 0.0


def test_red_flag_detection() -> None:
    """Agent should be scored on red flag detection."""
    task = load_task("easy/civic_2019")
    assert len(task.red_flags) > 0

    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=17500,
        turns_taken=6,
        red_flags_detected=["civic-004 priced suspiciously below market value"],
    )
    scored = compute_metrics(task, result)
    assert scored.red_flag_score is not None
    assert 0 < scored.red_flag_score <= 1.0


def test_red_flag_none_detected() -> None:
    """Agent that detects no red flags gets 0."""
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=17500,
        turns_taken=6,
        red_flags_detected=[],
    )
    scored = compute_metrics(task, result)
    assert scored.red_flag_score == 0.0


def test_composite_score_penalizes_poor_research() -> None:
    """Composite score should be lower when research is incomplete."""
    task = load_task("easy/civic_2019")

    good = compute_metrics(task, TaskResult(
        task_id="t1", deal_closed=True, final_price=17500, turns_taken=6,
        research_items_found=["fair market value", "vehicle history check", "recall status", "mileage verification"],
        red_flags_detected=list(task.red_flags),
    ))
    bad = compute_metrics(task, TaskResult(
        task_id="t2", deal_closed=True, final_price=17500, turns_taken=6,
        research_items_found=[],
        red_flags_detected=[],
    ))
    assert composite_score(good) > composite_score(bad)


def test_overpay_penalty_scales() -> None:
    """Overpay penalty should scale with how much over budget."""
    task = load_task("easy/civic_2019")

    slight = compute_metrics(task, TaskResult(
        task_id="t1", deal_closed=True, final_price=19200, turns_taken=6,
    ))
    big = compute_metrics(task, TaskResult(
        task_id="t2", deal_closed=True, final_price=22000, turns_taken=6,
    ))
    assert slight.overpay_penalty < big.overpay_penalty


def test_aggregate_results() -> None:
    task = load_task("easy/civic_2019")
    results = [
        compute_metrics(task, TaskResult(
            task_id="t1", deal_closed=True, final_price=17000, turns_taken=5,
        )),
        compute_metrics(task, TaskResult(
            task_id="t2", deal_closed=False,
        )),
    ]
    agg = aggregate_results("test-agent", "easy", results)
    assert agg.completion_rate == 0.5
    assert agg.mean_deal_score != 0.0
    assert agg.mean_composite_score != 0.0


def test_information_quality_partial() -> None:
    """Partial research should give partial score."""
    task = load_task("easy/civic_2019")
    result = TaskResult(
        task_id=task.task_id,
        deal_closed=True,
        final_price=17500,
        turns_taken=6,
        research_items_found=["fair market value"],
    )
    scored = compute_metrics(task, result)
    assert scored.information_quality is not None
    assert 0 < scored.information_quality < 1.0


def test_hard_task_tight_budget() -> None:
    """Hard task budget should be very close to minimum price."""
    task = load_task("hard/wrx_limited_2020")
    margin = task.budget_ceiling - task.seller.minimum_price
    assert margin <= 100  # Budget is basically at seller's minimum
    assert task.seller.firmness >= 0.85
    assert task.max_turns <= 20


def test_medium_task_has_hidden_issues() -> None:
    """Medium tasks should have hidden issues for agents to discover."""
    task = load_task("medium/accord_sport_2020")
    assert len(task.seller.hidden_issues) >= 2
    assert task.seller.firmness >= 0.7
