# Lowball: Can AI Agents Negotiate for a Car?

Lowball is a benchmark for evaluating foundation models and autonomous agents on their ability to **find, research, and negotiate the purchase of a common car** — a real-world, multi-step task that requires computer use, information synthesis, and adversarial human interaction.

## Motivation

Existing agent benchmarks evaluate code generation (SWE-bench), web navigation (WebArena), or desktop tasks (OSWorld). Lowball targets a gap: **high-stakes, multi-turn negotiation with humans**, grounded in a concrete consumer task that millions of people perform every year.

Buying a used car is hard because it requires:
- **Search & Discovery** — browsing listings across multiple platforms
- **Market Research** — understanding fair market value, vehicle history, and pricing signals
- **Strategic Communication** — contacting sellers, asking the right questions, and detecting red flags
- **Negotiation** — haggling for the best possible price using leverage, timing, and persuasion

## Benchmark Structure

Each task instance specifies a **target vehicle** (e.g., "2019 Honda Civic EX with under 60k miles") and a **budget ceiling**. The agent must:

1. **Source** — Find matching listings on simulated car marketplaces
2. **Research** — Determine fair market value using pricing tools and vehicle history
3. **Negotiate** — Engage with a simulated seller to reach the lowest possible price
4. **Close** — Make a final offer that is accepted by the seller

### Task Difficulty Tiers

| Tier | Description |
|------|-------------|
| **Easy** | Common car, cooperative seller, wide budget margin |
| **Medium** | Specific trim/options, moderately firm seller, tighter budget |
| **Hard** | Rare configuration, aggressive seller, minimal budget headroom |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `deal_score` | Final price relative to fair market value (lower is better) |
| `budget_adherence` | Whether the agent stayed within budget |
| `negotiation_efficiency` | Number of turns to reach agreement |
| `information_quality` | Did the agent surface key vehicle details (history, condition, recalls)? |
| `strategy_score` | LLM-judged quality of negotiation tactics |
| `completion_rate` | Fraction of tasks where a deal was successfully closed |

The **primary metric** is `deal_score`, defined as:

```
deal_score = (fair_market_value - final_price) / fair_market_value
```

A positive `deal_score` means the agent negotiated below market value. The benchmark reports mean deal score across all task instances.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Agent Under │────▶│  Browser Env     │────▶│  Simulated       │
│  Evaluation  │     │  (Marketplace)   │     │  Marketplace     │
│              │◀────│                  │◀────│  (Listings DB)   │
└─────────────┘     └──────────────────┘     └─────────────────┘
       │                                            │
       │            ┌──────────────────┐            │
       └───────────▶│  Messaging Env   │◀───────────┘
                    │  (Negotiation)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Simulated Seller │
                    │  (LLM Persona)   │
                    └──────────────────┘
```

## Quick Start

```bash
# Install
pip install lowball

# Run evaluation on a single task
lowball run --task easy/civic_2019 --agent openai:gpt-4o

# Run full benchmark suite
lowball run --suite easy --agent anthropic:claude-sonnet-4-6

# View results
lowball report --run-dir results/latest
```

## Project Structure

```
lowball/
├── lowball/
│   ├── tasks/           # Task schema and loader
│   ├── environment/     # Browser + messaging environments
│   ├── evaluation/      # Metrics, LLM judge, harness
│   └── simulation/      # Seller personas, marketplace sim
├── data/
│   ├── tasks/           # Task instance definitions (YAML)
│   ├── personas/        # Seller persona configs
│   └── vehicles/        # Vehicle database
└── configs/             # Runtime configuration
```

## Developing

```bash
git clone https://github.com/sam-silv/lowball.git
cd lowball
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@misc{lowball2026,
  title={Lowball: Evaluating AI Agents on Real-World Car Negotiation},
  year={2026}
}
```

## License

MIT
