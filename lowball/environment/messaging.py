"""Messaging environment for buyer-seller negotiation.

Manages structured tool-based negotiation between buyer agent and seller.
"""

from pydantic import BaseModel, Field

from lowball.simulation.seller import SimulatedSeller, NegotiationState


class MessageTurn(BaseModel):
    turn: int
    buyer_action: str  # "send_message", "make_offer", "walk_away"
    buyer_message: str
    buyer_price: float | None = None
    seller_action: str | None = None  # "counter", "accept", "reject"
    seller_message: str = ""
    seller_price: float | None = None


class MessagingEnvironment:
    """Manages the negotiation conversation between buyer agent and seller."""

    def __init__(self, seller: SimulatedSeller) -> None:
        self.seller = seller
        self.history: list[MessageTurn] = []
        self._buyer_walked_away = False

    @property
    def state(self) -> NegotiationState:
        return self.seller.state

    @property
    def is_complete(self) -> bool:
        return self.seller.state.deal_closed or self._rejected() or self._buyer_walked_away

    @property
    def deal_price(self) -> float | None:
        return self.seller.state.deal_price

    def handle_walk_away(self, reason: str) -> dict[str, object]:
        """Handle buyer walking away from negotiation."""
        self._buyer_walked_away = True
        self.history.append(MessageTurn(
            turn=len(self.history) + 1,
            buyer_action="walk_away",
            buyer_message=reason,
        ))
        return {"action": "ended", "message": "Buyer walked away."}

    async def send_message(self, message: str) -> dict[str, object]:
        """Send a regular message and get the seller's structured response."""
        if self.is_complete:
            return {"action": "ended", "message": "Negotiation has ended."}

        seller_response = await self.seller.respond(message)
        self.history.append(MessageTurn(
            turn=len(self.history) + 1,
            buyer_action="send_message",
            buyer_message=message,
            seller_action=str(seller_response["action"]),
            seller_message=str(seller_response["message"]),
            seller_price=seller_response.get("price"),  # type: ignore[arg-type]
        ))
        return seller_response

    async def make_offer(self, price: float, message: str) -> dict[str, object]:
        """Make a formal offer and get the seller's structured response."""
        if self.is_complete:
            return {"action": "ended", "message": "Negotiation has ended."}

        # Include the price in the message sent to the seller
        full_message = f"{message}\n\nMy offer: ${price:,.0f}"
        seller_response = await self.seller.respond(full_message)
        self.history.append(MessageTurn(
            turn=len(self.history) + 1,
            buyer_action="make_offer",
            buyer_message=message,
            buyer_price=price,
            seller_action=str(seller_response["action"]),
            seller_message=str(seller_response["message"]),
            seller_price=seller_response.get("price"),  # type: ignore[arg-type]
        ))
        return seller_response

    def _rejected(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        return last.seller_action == "reject"

    def get_transcript(self) -> list[dict[str, str]]:
        """Return the full conversation as a list of role/content dicts."""
        transcript: list[dict[str, str]] = []
        for turn in self.history:
            buyer_content = turn.buyer_message
            if turn.buyer_price is not None:
                buyer_content += f"\n\n[OFFER: ${turn.buyer_price:,.0f}]"
            transcript.append({"role": "buyer", "content": buyer_content})

            if turn.seller_message:
                seller_content = turn.seller_message
                if turn.seller_price is not None:
                    seller_content += f"\n\n[{turn.seller_action.upper()}: ${turn.seller_price:,.0f}]"  # type: ignore[union-attr]
                elif turn.seller_action == "reject":
                    seller_content += "\n\n[REJECTED]"
                transcript.append({"role": "seller", "content": seller_content})
        return transcript
