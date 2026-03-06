"""Base agent with shared browsing and negotiation logic."""

from abc import ABC, abstractmethod

from lowball.environment.messaging import MessagingEnvironment
from lowball.environment.tools import MarketplaceToolDispatcher


BROWSE_SYSTEM_PROMPT = """\
You are a car buyer agent. Your job is to find the best matching vehicle \
on a marketplace, evaluate it, and select it for negotiation.

You will be given a buying goal describing the target vehicle, budget, \
and requirements. Use the search and listing tools to browse the \
marketplace, compare listings, and select the best match.

Be thorough: search broadly first, then inspect individual listings. \
Compare prices, note mileage and condition. Pick the listing that best \
matches the requirements AND offers the most negotiation room (e.g., \
higher days listed, higher price relative to comparable listings).

IMPORTANT — Watch for red flags and avoid suspicious listings:
- Prices significantly below market value (possible scam or salvage title)
- Missing vehicle history reports
- High number of previous owners on low-mileage vehicles
- Vague descriptions that avoid specifics about condition or history
- "Fresh paint" or "new floor mats" on used cars (may hide damage)
- Modifications on vehicles that should be stock
- Condition listed as "fair" but mileage seems impossibly low

When you've found the best listing, call the select_listing tool. You MUST \
include detailed research_findings and red_flags arrays. Your research \
findings should cover:
- Fair market value (cite specific price ranges from comparable listings)
- Vehicle history (clean title, accident history, number of owners)
- Mileage verification (is it reasonable for the vehicle's age?)
- Any recall information or known issues for the model
- Seller motivation and listing age (days on market, urgency signals)
- Any red flags from OTHER listings you reviewed and rejected
"""

NEGOTIATE_SYSTEM_PROMPT = """\
You are an expert car buyer negotiating to get the lowest possible price. \
You are buying on behalf of a client.

You have three tools available:
- send_message: Send a regular message (questions, rapport, discussion)
- make_offer: Make a formal price offer (the seller will accept, reject, or counter)
- walk_away: End negotiation without a deal

STRATEGY GUIDELINES:
1. Open with rapport — compliment the car, ask about its history, show \
   genuine interest before discussing price. Use send_message for this.
2. Anchor low — your first offer should be 15-25% below asking price, \
   justified by market research, comparable listings, or minor issues. \
   Use make_offer with a specific price.
3. Use market data — reference fair market value, similar listings at \
   lower prices, and any issues you discovered during research.
4. Concede slowly — increase your offer in small increments ($200-500), \
   signaling that you're near your limit.
5. Create urgency — mention you're looking at other cars if the seller \
   seems firm.
6. Ask about flaws — probe for hidden issues (maintenance history, \
   accidents, mechanical problems) to gain leverage.
7. Be willing to walk away — if the price is above your budget, \
   use the walk_away tool.
8. Close decisively — when the price is right, make your best offer \
   quickly before the seller reconsiders.

RULES:
- Never reveal your actual budget ceiling.
- Never accept the first price offered.
- Always justify your offers with specific reasons.
- Stay polite and professional throughout.
- Use make_offer for ALL price proposals — do not mention prices in send_message.
"""


class BaseAgent(ABC):
    """Abstract base agent with shared prompt construction."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.task_description: str = ""
        self.tools: MarketplaceToolDispatcher | None = None
        self.research_findings: list[str] = []
        self.red_flags_found: list[str] = []

    async def setup(self, task_description: str, tools: MarketplaceToolDispatcher) -> None:
        self.task_description = task_description
        self.tools = tools
        self.research_findings = []
        self.red_flags_found = []

    async def report_research(self) -> list[str]:
        return self.research_findings

    async def report_red_flags(self) -> list[str]:
        return self.red_flags_found

    @abstractmethod
    async def browse_and_select(self) -> str | None: ...

    @abstractmethod
    async def negotiate(self, messaging: MessagingEnvironment, max_turns: int) -> None:
        """Run the full negotiation loop using tool calls against the messaging environment."""
        ...

    def _build_negotiation_context(self) -> str:
        """Build the context string for negotiation."""
        return (
            f"BUYING GOAL:\n{self.task_description}\n\n"
            f"RESEARCH FINDINGS:\n"
            + "\n".join(f"- {f}" for f in self.research_findings)
            + "\n\nNegotiate with the seller using the available tools."
        )
