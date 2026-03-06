"""Tests for task loading and schema validation."""

import pytest

from lowball.tasks.loader import load_task, load_suite
from lowball.tasks.schema import Difficulty, TaskInstance


def test_load_easy_civic() -> None:
    task = load_task("easy/civic_2019")
    assert task.task_id == "easy/civic_2019"
    assert task.difficulty == Difficulty.EASY
    assert task.vehicle.make == "Honda"
    assert task.vehicle.model == "Civic"
    assert task.fair_market_value == 18200
    assert task.budget_ceiling == 19000
    assert task.seller.asking_price == 19500
    assert task.seller.minimum_price == 17500
    assert task.seller.firmness >= 0.4  # Not a pushover anymore
    assert task.max_turns <= 12
    assert len(task.required_research) >= 4
    assert len(task.red_flags) >= 2


def test_load_medium_accord() -> None:
    task = load_task("medium/accord_sport_2020")
    assert task.difficulty == Difficulty.MEDIUM
    assert task.vehicle.trim == "Sport"
    assert "leather seats" in task.vehicle.required_features
    assert task.seller.firmness >= 0.7
    assert len(task.seller.hidden_issues) >= 2
    assert task.max_turns <= 15


def test_load_hard_wrx() -> None:
    task = load_task("hard/wrx_limited_2020")
    assert task.difficulty == Difficulty.HARD
    assert task.seller.firmness >= 0.85
    assert len(task.seller.hidden_issues) >= 3
    assert len(task.required_research) >= 8
    assert task.max_turns <= 20
    # Budget should be extremely tight
    assert task.budget_ceiling - task.seller.minimum_price <= 100


def test_load_medium_deceptive() -> None:
    task = load_task("medium/civic_deceptive_2020")
    assert task.difficulty == Difficulty.MEDIUM
    assert task.seller.negotiation_style == "deceptive"
    assert len(task.seller.hidden_issues) >= 3
    assert len(task.red_flags) >= 3


def test_load_hard_walkaway() -> None:
    task = load_task("hard/accord_walkaway_2021")
    assert task.difficulty == Difficulty.HARD
    assert task.max_turns <= 10  # Very few turns to close
    assert task.seller.firmness >= 0.8


def test_load_suite_easy() -> None:
    tasks = load_suite("easy")
    assert len(tasks) >= 2
    assert all(t.difficulty == Difficulty.EASY for t in tasks)


def test_load_suite_medium() -> None:
    tasks = load_suite("medium")
    assert len(tasks) >= 2
    assert all(t.difficulty == Difficulty.MEDIUM for t in tasks)


def test_load_suite_hard() -> None:
    tasks = load_suite("hard")
    assert len(tasks) >= 2
    assert all(t.difficulty == Difficulty.HARD for t in tasks)


def test_load_missing_task() -> None:
    with pytest.raises(FileNotFoundError):
        load_task("nonexistent/fake_car")


def test_all_tasks_have_red_flags_or_research() -> None:
    """Every task should require meaningful research."""
    for suite in ["easy", "medium", "hard"]:
        tasks = load_suite(suite)
        for task in tasks:
            assert len(task.required_research) >= 3, (
                f"Task {task.task_id} has too few research requirements"
            )


def test_budget_headroom_decreases_with_difficulty() -> None:
    """Harder tasks should have tighter budgets relative to seller minimum."""
    easy = load_task("easy/civic_2019")
    hard = load_task("hard/wrx_limited_2020")

    easy_margin = (easy.budget_ceiling - easy.seller.minimum_price) / easy.seller.minimum_price
    hard_margin = (hard.budget_ceiling - hard.seller.minimum_price) / hard.seller.minimum_price

    assert easy_margin > hard_margin
