"""Simulated car seller powered by an LLM with a persona."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from lowball.tasks.schema import SellerConfig

PERSONAS_DIR = Path(__file__).parent.parent.parent / "data" / "personas"


class SellerPersona(BaseModel):
    """A seller's personality and background."""

    persona_id: str
    name: str
    background: str = Field(description="Short backstory for the seller")
    communication_style: str = Field(description="How they talk: formal, casual, terse, etc.")
    tactics: list[str] = Field(
        default_factory=list,
        description="Negotiation tactics this seller uses",
    )
    system_prompt: str = Field(description="System prompt for the LLM playing this seller")


def load_persona(persona_id: str) -> SellerPersona:
    persona_path = PERSONAS_DIR / f"{persona_id}.yaml"
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona not found: {persona_path}")
    with open(persona_path) as f:
        raw = yaml.safe_load(f)
    raw["persona_id"] = persona_id
    return SellerPersona(**raw)


class NegotiationMessage(BaseModel):
    role: str = Field(description="'buyer' or 'seller'")
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class NegotiationState(BaseModel):
    """Tracks the state of an ongoing negotiation."""

    messages: list[NegotiationMessage] = Field(default_factory=list)
    current_offer: float | None = None
    seller_last_price: float | None = None
    deal_closed: bool = False
    deal_price: float | None = None
    turn_count: int = 0


class SimulatedSeller:
    """An LLM-powered simulated car seller.

    The seller follows its persona and config to respond to buyer messages,
    make counteroffers, and eventually accept or reject deals.
    """

    def __init__(self, config: SellerConfig, llm_client: object | None = None) -> None:
        self.config = config
        self.persona = load_persona(config.persona_id)
        self.state = NegotiationState()
        self.llm_client = llm_client
        self._system_prompt = self.build_system_prompt()

    def build_system_prompt(self) -> str:
        return f"""{self.persona.system_prompt}

SELLER PARAMETERS (hidden from buyer):
- Asking price: ${self.config.asking_price:,.0f}
- Minimum acceptable price: ${self.config.minimum_price:,.0f}
- Firmness: {self.config.firmness} (0=flexible, 1=firm)
- Urgency to sell: {self.config.urgency} (0=no rush, 1=desperate)
- Negotiation style: {self.config.negotiation_style}
- Hidden issues: {', '.join(self.config.hidden_issues) or 'None'}

RULES:
1. Never reveal your minimum price, firmness level, or urgency to sell.
2. Start near your asking price and concede slowly based on firmness.
3. Do NOT accept offers at your minimum price immediately — push back at least once or twice
   even if the offer is reasonable. Make the buyer work for every dollar.
4. If the buyer's offer is below minimum, push back firmly and consider walking away.
5. If the buyer immediately offers a high price or seems eager, RAISE your expectations
   and concede less — they clearly have budget room.
6. Use your persona's tactics naturally in conversation.
7. Ask probing questions to understand the buyer's budget and timeline before conceding.
8. If the buyer reveals their budget or maximum price, anchor your counteroffers near that number.
9. Use emotional anchoring: mention how much you love the car, what you spent on maintenance, etc.
10. If the buyer hasn't mentioned specific vehicle concerns or market data, don't volunteer concessions.
11. Make counteroffers in small decrements ($100-300) to signal firmness.
12. You should reject and walk away if the buyer is rude, too aggressive, or wastes your time.

RESPONSE FORMAT — respond with EXACTLY one of these JSON structures, no other text:

For ALL regular responses, counteroffers, and continuing negotiation (even if the offer is low):
{{"action": "counter", "price": <your_counter_price>, "message": "<your message>"}}

ONLY when you agree to the buyer's price and want to close the deal:
{{"action": "accept", "price": <agreed_price>, "message": "<your acceptance message>"}}

ONLY when you want to permanently end the negotiation and walk away (buyer is rude, insulting, or wasting your time):
{{"action": "reject", "message": "<your rejection message>"}}

IMPORTANT: Use "counter" for the vast majority of responses. A low offer is NOT a reason to reject — \
it's a reason to counter. Only use "reject" if the buyer is genuinely offensive or you've exhausted \
all patience after many rounds. Good sellers keep negotiating.
"""

    async def respond(self, buyer_message: str) -> dict[str, object]:
        """Generate a structured seller response to a buyer message.

        Returns a dict with keys: action ("counter"|"accept"|"reject"),
        price (float|None), message (str).
        """
        import json as _json

        self.state.messages.append(NegotiationMessage(role="buyer", content=buyer_message))
        self.state.turn_count += 1

        # Build message history for LLM
        llm_messages = [{"role": "system", "content": self._system_prompt}]
        for msg in self.state.messages:
            llm_messages.append({
                "role": "user" if msg.role == "buyer" else "assistant",
                "content": msg.content,
            })

        # Call LLM and parse structured response
        response_text = await self._call_llm(llm_messages)
        parsed = self._parse_response(response_text)

        if parsed["action"] == "accept":
            self.state.deal_closed = True
            self.state.deal_price = float(parsed["price"])  # type: ignore[arg-type]

        # Store the message text for transcript
        self.state.messages.append(NegotiationMessage(
            role="seller",
            content=str(parsed["message"]),
            metadata={"action": parsed["action"], "price": parsed.get("price")},
        ))
        return parsed

    def _parse_response(self, text: str) -> dict[str, object]:
        """Parse the seller LLM's JSON response, with fallback for malformed output."""
        import json as _json

        # Try direct JSON parse
        try:
            parsed = _json.loads(text)
            if "action" in parsed and "message" in parsed:
                return parsed
        except _json.JSONDecodeError:
            pass

        # Fallback: extract JSON from text that may have extra content
        import re
        match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+?"[^{}]*\}', text)
        if match:
            try:
                parsed = _json.loads(match.group())
                if "action" in parsed and "message" in parsed:
                    return parsed
            except _json.JSONDecodeError:
                pass

        # Last resort: treat as free-text counter
        return {"action": "counter", "price": None, "message": text}

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM backing the seller. Override for different providers."""
        if self.llm_client is None:
            raise RuntimeError(
                "No LLM client configured for SimulatedSeller. "
                "Pass an llm_client or subclass and override _call_llm."
            )
        # Default implementation using OpenAI-compatible client
        response = await self.llm_client.chat.completions.create(  # type: ignore[union-attr]
            model="gpt-5.4",
            messages=messages,
            temperature=0.8,
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content  # type: ignore[union-attr]
