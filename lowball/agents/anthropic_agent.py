"""Anthropic Claude agent using tool use for browsing and negotiation."""

import json

import anthropic

from lowball.agents.base import BaseAgent, BROWSE_SYSTEM_PROMPT, NEGOTIATE_SYSTEM_PROMPT
from lowball.environment.messaging import MessagingEnvironment
from lowball.environment.tools import anthropic_tool_schemas, anthropic_negotiation_tool_schemas


class AnthropicAgent(BaseAgent):
    """Agent using Claude's messages API with tool use for marketplace
    browsing and negotiation.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        super().__init__(model)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def browse_and_select(self) -> str | None:
        """Use tool use to browse the marketplace and select a listing."""
        tools = anthropic_tool_schemas()
        messages: list[dict] = [
            {
                "role": "user",
                "content": f"GOAL: {self.task_description}\n\nBrowse the marketplace and find the best listing.",
            },
        ]

        max_steps = 20
        for step in range(max_steps):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=BROWSE_SYSTEM_PROMPT,
                tools=tools,  # type: ignore[arg-type]
                messages=messages,
                temperature=0.3,
            )

            # Append assistant response
            messages.append({"role": "assistant", "content": response.content})

            # Check if model used any tools
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            # Process each tool use and build tool_result message
            tool_results = []
            selected_id: str | None = None

            for block in tool_uses:
                name = block.name
                args = block.input
                print(f"  [browse step {step}] {name}({args})")

                result = self.tools.dispatch(name, args)  # type: ignore[union-attr]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

                # Handle select_listing
                if name == "select_listing":
                    self.research_findings.extend(args.get("research_findings", []))
                    self.red_flags_found.extend(args.get("red_flags", []))
                    # Fallback: extract research from text blocks if not in tool args
                    if not args.get("research_findings"):
                        text_blocks = [b.text for b in response.content if b.type == "text"]
                        if text_blocks:
                            self.research_findings.extend(text_blocks)
                    selected_id = args["listing_id"]

            messages.append({"role": "user", "content": tool_results})

            if selected_id:
                return selected_id

        return None

    async def negotiate(self, messaging: MessagingEnvironment, max_turns: int) -> None:
        """Run negotiation loop using Claude tool use."""
        tools = anthropic_negotiation_tool_schemas()
        messages: list[dict] = [
            {"role": "user", "content": self._build_negotiation_context()},
        ]

        for turn in range(max_turns):
            if messaging.is_complete:
                break

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=NEGOTIATE_SYSTEM_PROMPT,
                tools=tools,  # type: ignore[arg-type]
                messages=messages,
                temperature=0.7,
            )

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            tool_results = []
            done = False

            for block in tool_uses:
                name = block.name
                args = block.input
                print(f"  [negotiate turn {turn}] {name}({args})")

                if name == "walk_away":
                    messaging.handle_walk_away(args["reason"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"status": "walked_away"}),
                    })
                    done = True
                    break

                if name == "make_offer":
                    message = args.get("message", f"I'd like to offer ${args['price']:,.0f}.")
                    seller_response = await messaging.make_offer(args["price"], message)
                elif name == "send_message":
                    seller_response = await messaging.send_message(args["message"])
                else:
                    seller_response = {"action": "error", "message": f"Unknown tool: {name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(seller_response),
                })

            messages.append({"role": "user", "content": tool_results})

            if done:
                return
