"""CLI entry point for Lowball."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import click
from rich.console import Console
from rich.table import Table

from lowball.tasks.loader import load_suite, load_task

console = Console()


def _create_agent(agent_spec: str):
    """Parse an agent spec like 'anthropic:claude-sonnet-4-6' or 'openai:gpt-4o'
    and return a configured agent instance."""
    if ":" not in agent_spec:
        raise click.BadParameter(
            f"Agent must be 'provider:model' (e.g., 'anthropic:claude-sonnet-4-6'). Got: {agent_spec}"
        )

    provider, model = agent_spec.split(":", 1)

    if provider == "anthropic":
        from lowball.agents.anthropic_agent import AnthropicAgent
        return AnthropicAgent(model=model)
    elif provider == "openai":
        from lowball.agents.openai_agent import OpenAIAgent
        return OpenAIAgent(model=model)
    else:
        raise click.BadParameter(f"Unknown provider '{provider}'. Supported: anthropic, openai")


def _create_seller_llm_client():
    """Create an OpenAI client for powering the simulated seller."""
    import openai
    return openai.AsyncOpenAI()


@click.group()
def main() -> None:
    """Lowball: Evaluate AI agents on car negotiation tasks."""


@main.command()
@click.option("--task", "task_id", help="Single task ID (e.g., easy/civic_2019)")
@click.option("--suite", help="Run all tasks in a suite (easy, medium, hard)")
@click.option("--agent", required=True, help="Agent identifier (e.g., openai:gpt-4o)")
@click.option("--output", default="results", help="Output directory for results")
def run(task_id: str | None, suite: str | None, agent: str, output: str) -> None:
    """Run evaluation tasks against an agent."""
    if not task_id and not suite:
        console.print("[red]Specify either --task or --suite[/red]")
        raise SystemExit(1)

    agent_instance = _create_agent(agent)
    seller_llm = _create_seller_llm_client()

    from lowball.evaluation.harness import EvaluationHarness

    harness = EvaluationHarness(
        agent=agent_instance,
        llm_client=seller_llm,
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if task_id:
        task = load_task(task_id)
        console.print(f"[bold]Task:[/bold] {task.task_id} ({task.difficulty.value})")
        console.print(f"  Vehicle: {task.vehicle.year_min}-{task.vehicle.year_max} "
                       f"{task.vehicle.make} {task.vehicle.model}")
        console.print(f"  Budget: ${task.budget_ceiling:,.0f}  |  FMV: ${task.fair_market_value:,.0f}")
        console.print(f"[bold]Agent:[/bold] {agent}\n")

        result = asyncio.run(harness.run_task(task))

        _print_task_result(result)
        _save_results(output_dir, {"task_results": [result.model_dump()]})

    elif suite:
        tasks = load_suite(suite)
        console.print(f"[bold]Suite:[/bold] {suite} ({len(tasks)} tasks)")
        console.print(f"[bold]Agent:[/bold] {agent}\n")

        benchmark = asyncio.run(harness.run_suite(tasks, agent_id=agent, suite=suite))

        for result in benchmark.task_results:
            _print_task_result(result)
            console.print()

        _print_benchmark_summary(benchmark)
        _save_results(output_dir, benchmark.model_dump())


def _print_task_result(result) -> None:
    """Print a single task result."""
    status = "[green]DEAL CLOSED[/green]" if result.deal_closed else "[red]NO DEAL[/red]"
    console.print(f"  {result.task_id}: {status}")
    if result.final_price is not None:
        console.print(f"    Final price: ${result.final_price:,.0f}")
    console.print(f"    Turns: {result.turns_taken}  |  Time: {result.time_elapsed_seconds:.1f}s")
    if result.deal_score is not None:
        console.print(f"    Deal score: {result.deal_score:.3f}")
    if result.information_quality is not None:
        console.print(f"    Info quality: {result.information_quality:.1%}")


def _print_benchmark_summary(benchmark) -> None:
    """Print aggregate benchmark results."""
    table = Table(title=f"Lowball Results — {benchmark.agent_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Completion Rate", f"{benchmark.completion_rate:.1%}")
    table.add_row("Mean Deal Score", f"{benchmark.mean_deal_score:.3f}")
    table.add_row("Budget Adherence", f"{benchmark.budget_adherence_rate:.1%}")
    table.add_row("Negotiation Efficiency", f"{benchmark.mean_negotiation_efficiency:.3f}")
    table.add_row("Information Quality", f"{benchmark.mean_information_quality:.3f}")
    table.add_row("Strategy Score", f"{benchmark.mean_strategy_score:.3f}")
    table.add_row("Red Flag Detection", f"{benchmark.mean_red_flag_score:.3f}")
    table.add_row("Composite Score", f"{benchmark.mean_composite_score:.3f}")

    console.print(table)


def _save_results(output_dir: Path, data: dict) -> None:
    """Save results to JSON."""
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved to {results_path}[/dim]")


@main.command()
@click.option("--run-dir", required=True, help="Path to results directory")
def report(run_dir: str) -> None:
    """Display results from a completed evaluation run."""
    results_path = Path(run_dir) / "results.json"
    if not results_path.exists():
        console.print(f"[red]No results found at {results_path}[/red]")
        raise SystemExit(1)

    with open(results_path) as f:
        data = json.load(f)

    table = Table(title=f"Lowball Results — {data.get('agent_id', 'unknown')}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Completion Rate", f"{data.get('completion_rate', 0):.1%}")
    table.add_row("Mean Deal Score", f"{data.get('mean_deal_score', 0):.3f}")
    table.add_row("Budget Adherence", f"{data.get('budget_adherence_rate', 0):.1%}")
    table.add_row("Negotiation Efficiency", f"{data.get('mean_negotiation_efficiency', 0):.3f}")
    table.add_row("Information Quality", f"{data.get('mean_information_quality', 0):.3f}")
    table.add_row("Strategy Score", f"{data.get('mean_strategy_score', 0):.3f}")
    table.add_row("Red Flag Detection", f"{data.get('mean_red_flag_score', 0):.3f}")
    table.add_row("Composite Score", f"{data.get('mean_composite_score', 0):.3f}")

    console.print(table)


@main.command()
def list_tasks() -> None:
    """List all available tasks."""
    for suite in ["easy", "medium", "hard"]:
        try:
            tasks = load_suite(suite)
            console.print(f"\n[bold]{suite.upper()}[/bold] ({len(tasks)} tasks)")
            for task in tasks:
                console.print(f"  {task.task_id}: {task.description}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
