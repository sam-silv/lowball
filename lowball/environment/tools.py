"""Marketplace tools for agent browsing via function calling.

Provides structured tool definitions and a dispatcher that agents use
to search listings, inspect details, and select a listing to negotiate on.
"""

import json

from lowball.simulation.market import Marketplace


# Tool schemas (JSON Schema format, compatible with OpenAI and Anthropic)
MARKETPLACE_TOOLS = [
    {
        "name": "search_listings",
        "description": (
            "Search the car marketplace for vehicle listings. "
            "Returns a summary list with listing_id, year, make, model, "
            "trim, mileage, price, condition, and location. "
            "Call with no arguments to see all listings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "make": {"type": "string", "description": "Filter by make (e.g. Honda, Toyota)"},
                "model": {"type": "string", "description": "Filter by model (e.g. Civic, Corolla)"},
                "year_min": {"type": "integer", "description": "Minimum model year"},
                "year_max": {"type": "integer", "description": "Maximum model year"},
                "price_max": {"type": "number", "description": "Maximum listing price"},
                "mileage_max": {"type": "integer", "description": "Maximum mileage"},
            },
        },
    },
    {
        "name": "get_listing_details",
        "description": (
            "Get full details for a specific listing including description, "
            "features, seller name, days listed, number of views, vehicle "
            "history availability, accident history, and number of previous owners."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "listing_id": {"type": "string", "description": "The listing ID to look up"},
            },
            "required": ["listing_id"],
        },
    },
    {
        "name": "select_listing",
        "description": (
            "Select a listing to negotiate on. Call this when you've found "
            "the best match. Include any research findings and red flags "
            "you discovered during browsing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "listing_id": {"type": "string", "description": "The listing ID to negotiate on"},
                "research_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Research findings discovered during browsing",
                },
                "red_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Red flags or suspicious listings detected",
                },
            },
            "required": ["listing_id", "research_findings", "red_flags"],
        },
    },
]


NEGOTIATION_TOOLS = [
    {
        "name": "send_message",
        "description": (
            "Send a message to the seller during negotiation. Use this for "
            "general conversation, questions, rapport-building, and discussion. "
            "Do NOT include a price offer in this message — use make_offer for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Your message to the seller"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "make_offer",
        "description": (
            "Make a formal price offer to the seller. The seller will respond "
            "with accept, reject, or counter. Include a persuasive message "
            "justifying your offer price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "price": {"type": "number", "description": "Your offer price in dollars"},
                "message": {"type": "string", "description": "Message accompanying your offer"},
            },
            "required": ["price", "message"],
        },
    },
    {
        "name": "walk_away",
        "description": (
            "End the negotiation without making a deal. Use this if the seller's "
            "price is too high and you cannot reach agreement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why you're walking away"},
            },
            "required": ["reason"],
        },
    },
]


def openai_negotiation_tool_schemas() -> list[dict]:
    """Return negotiation tool schemas in OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in NEGOTIATION_TOOLS
    ]


def anthropic_negotiation_tool_schemas() -> list[dict]:
    """Return negotiation tool schemas in Anthropic tool use format."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in NEGOTIATION_TOOLS
    ]


def openai_tool_schemas() -> list[dict]:
    """Return tool schemas in OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in MARKETPLACE_TOOLS
    ]


def anthropic_tool_schemas() -> list[dict]:
    """Return tool schemas in Anthropic tool use format."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in MARKETPLACE_TOOLS
    ]


class MarketplaceToolDispatcher:
    """Executes marketplace tool calls and returns JSON results."""

    def __init__(self, marketplace: Marketplace) -> None:
        self.marketplace = marketplace

    def dispatch(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the JSON result string."""
        if tool_name == "search_listings":
            return self._search(arguments)
        elif tool_name == "get_listing_details":
            return self._get_details(arguments)
        elif tool_name == "select_listing":
            return self._select(arguments)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _search(self, args: dict) -> str:
        results = self.marketplace.search(
            make=args.get("make"),
            model=args.get("model"),
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
            price_max=args.get("price_max"),
            mileage_max=args.get("mileage_max"),
        )
        summaries = [
            {
                "listing_id": l.listing_id,
                "year": l.year,
                "make": l.make,
                "model": l.model,
                "trim": l.trim,
                "mileage": l.mileage,
                "price": l.price,
                "condition": l.condition,
                "location": l.location,
            }
            for l in results
        ]
        return json.dumps(summaries)

    def _get_details(self, args: dict) -> str:
        listing = self.marketplace.get_listing(args["listing_id"])
        if not listing:
            return json.dumps({"error": f"Listing not found: {args['listing_id']}"})
        return json.dumps(listing.model_dump())

    def _select(self, args: dict) -> str:
        listing_id = args["listing_id"]
        listing = self.marketplace.get_listing(listing_id)
        if not listing:
            return json.dumps({"error": f"Listing not found: {listing_id}"})
        return json.dumps({"selected": listing_id, "confirmed": True})
