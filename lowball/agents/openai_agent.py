"""OpenAI agent using function calling for browsing and negotiation."""

import json

import openai

from lowball.agents.base import BaseAgent, BROWSE_SYSTEM_PROMPT, NEGOTIATE_SYSTEM_PROMPT
from lowball.environment.messaging import MessagingEnvironment
from lowball.environment.tools import openai_tool_schemas, openai_negotiation_tool_schemas


class OpenAIAgent(BaseAgent):
    """Agent using OpenAI's chat completions with function calling
    for marketplace browsing and negotiation.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        super().__init__(model)
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def browse_and_select(self) -> str | None:
        """Use function calling to browse the marketplace and select a listing."""
        tools = openai_tool_schemas()
        messages: list[dict] = [
            {"role": "system", "content": BROWSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"GOAL: {self.task_description}\n\nBrowse the marketplace and find the best listing."},
        ]

        max_steps = 20
        for step in range(max_steps):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                temperature=0.3,
            )

            choice = response.choices[0]
            messages.append(choice.message.model_dump(exclude_none=True))  # type: ignore[arg-type]

            # No tool calls — model is done
            if not choice.message.tool_calls:
                break

            # Process each tool call
            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  [browse step {step}] {name}({args})")

                result = self.tools.dispatch(name, args)  # type: ignore[union-attr]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                # Handle select_listing — capture findings and return
                if name == "select_listing":
                    self.research_findings.extend(args.get("research_findings", []))
                    self.red_flags_found.extend(args.get("red_flags", []))
                    return args["listing_id"]

        return None

    async def negotiate(self, messaging: MessagingEnvironment, max_turns: int) -> None:
        """Run negotiation loop using function calling tools."""
        tools = openai_negotiation_tool_schemas()
        messages: list[dict] = [
            {"role": "system", "content": NEGOTIATE_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_negotiation_context()},
        ]

        for turn in range(max_turns):
            if messaging.is_complete:
                break

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                temperature=0.7,
            )

            choice = response.choices[0]
            messages.append(choice.message.model_dump(exclude_none=True))  # type: ignore[arg-type]

            if not choice.message.tool_calls:
                break

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  [negotiate turn {turn}] {name}({args})")

                if name == "walk_away":
                    messaging.handle_walk_away(args["reason"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"status": "walked_away"}),
                    })
                    return

                if name == "make_offer":
                    seller_response = await messaging.make_offer(args["price"], args["message"])
                elif name == "send_message":
                    seller_response = await messaging.send_message(args["message"])
                else:
                    seller_response = {"action": "error", "message": f"Unknown tool: {name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(seller_response),
                })
