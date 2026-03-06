"""LLM-based judge for evaluating negotiation strategy and research quality.

Uses separate LLM calls to assess the agent's negotiation tactics
and research thoroughness, similar to how MT-Bench uses LLM-as-judge.
"""

JUDGE_SYSTEM_PROMPT = """You are a ruthlessly honest expert negotiation coach evaluating a car buying negotiation.
You will be given the full transcript of a negotiation between a buyer (AI agent) and a seller.

Be STRICT. Most negotiations are mediocre. A score of 5 is average. Only truly exceptional
tactics earn 8+. Common AI agent failures to penalize heavily:

- Revealing budget or maximum price (immediate 1-2 on opening strategy)
- Accepting the first counteroffer or caving quickly (1-3 on tactical execution)
- Not referencing specific market data or comparables (1-3 on market knowledge)
- Making large concession jumps (>$500 at once) (1-3 on concession pattern)
- Being too polite/agreeable instead of firm (penalize tactical execution)
- Not asking about vehicle condition, history, or issues (penalize market knowledge)
- Failing to create any leverage or walk-away credibility (penalize tactical execution)

Evaluate the buyer's negotiation strategy on a scale of 1-10 across these dimensions:

1. **Opening Strategy** (1-10): Did the buyer build rapport WITHOUT revealing budget?
   Did they ask probing questions about the car's condition, history, and seller's
   motivation before discussing price? Did they anchor low with justification?

2. **Market Knowledge** (1-10): Did the buyer cite specific fair market values,
   comparable listings, vehicle-specific known issues, or recall data? Vague
   references to "market value" without numbers score 3 or below.

3. **Tactical Execution** (1-10): Did the buyer anchor low, make justified counteroffers,
   use silence/patience, create walk-away credibility, and leverage specific vehicle
   concerns? Did they avoid common traps like accepting urgency framing from the seller?

4. **Concession Pattern** (1-10): Did the buyer make progressively smaller concessions
   (signaling a firm floor)? Or did they make large, erratic jumps? Did they give
   concessions without getting something in return?

5. **Final Outcome** (1-10): Given the context, did the buyer achieve a genuinely
   good price? Paying at or above FMV is a 4 or below regardless of other factors.

6. **Red Flag Awareness** (1-10): Did the buyer ask about or surface potential issues
   with the vehicle? Did they notice inconsistencies in the seller's claims? Did they
   request documentation or verification?

Respond in this exact JSON format:
{
  "opening_strategy": <score>,
  "market_knowledge": <score>,
  "tactical_execution": <score>,
  "concession_pattern": <score>,
  "final_outcome": <score>,
  "red_flag_awareness": <score>,
  "overall_score": <weighted_average>,
  "reasoning": "<2-3 sentence explanation of key strengths and failures>"
}
"""


RESEARCH_JUDGE_PROMPT = """You are evaluating whether an AI car-buying agent's research findings \
cover the required research items and whether it detected the known red flags.

You will be given:
1. The agent's research findings (free-text list)
2. The agent's detected red flags (free-text list)
3. The required research items (short labels like "fair market value")
4. The known red flags in the marketplace (descriptions of suspicious listings)

For each required research item, determine if the agent's findings demonstrate \
that the agent actually investigated this topic. Be generous with matching — \
the agent's wording won't match exactly. For example:
- "fair market value" is covered if the agent mentions comparable prices, market range, or KBB values
- "vehicle history check" is covered if the agent mentions checking Carfax, clean history, or accident records
- "recall status" is covered if the agent mentions checking for recalls
- "mileage verification" is covered if the agent discusses the odometer reading or mileage relative to age
- "number of previous owners" is covered if the agent mentions owner count

For each known red flag, determine if the agent flagged it or a substantially similar concern.

Respond in this exact JSON format:
{
  "research_coverage": [
    {"item": "<required item>", "covered": true/false, "evidence": "<brief quote or null>"}
  ],
  "red_flag_coverage": [
    {"flag": "<known flag>", "detected": true/false, "evidence": "<brief quote or null>"}
  ],
  "research_score": <float 0.0 to 1.0>,
  "red_flag_score": <float 0.0 to 1.0>
}
"""


def build_judge_prompt(transcript: list[dict[str, str]], context: str) -> list[dict[str, str]]:
    """Build the prompt for the LLM judge."""
    conversation = "\n".join(
        f"{'BUYER' if t['role'] == 'buyer' else 'SELLER'}: {t['content']}"
        for t in transcript
    )

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nTranscript:\n{conversation}"},
    ]


async def judge_negotiation(
    transcript: list[dict[str, str]],
    context: str,
    llm_client: object,
) -> dict[str, object]:
    """Run the LLM judge on a negotiation transcript. Returns scores dict."""
    import json

    messages = build_judge_prompt(transcript, context)

    response = await llm_client.chat.completions.create(  # type: ignore[union-attr]
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)  # type: ignore[union-attr]


async def judge_research(
    research_findings: list[str],
    red_flags_detected: list[str],
    required_research: list[str],
    known_red_flags: list[str],
    llm_client: object,
) -> dict[str, object]:
    """Use LLM judge to evaluate research coverage and red flag detection."""
    import json

    user_content = (
        f"AGENT'S RESEARCH FINDINGS:\n"
        + "\n".join(f"- {f}" for f in research_findings) + "\n\n"
        f"AGENT'S DETECTED RED FLAGS:\n"
        + ("\n".join(f"- {f}" for f in red_flags_detected) if red_flags_detected else "(none)") + "\n\n"
        f"REQUIRED RESEARCH ITEMS:\n"
        + "\n".join(f"- {r}" for r in required_research) + "\n\n"
        f"KNOWN RED FLAGS:\n"
        + ("\n".join(f"- {f}" for f in known_red_flags) if known_red_flags else "(none)")
    )

    messages = [
        {"role": "system", "content": RESEARCH_JUDGE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = await llm_client.chat.completions.create(  # type: ignore[union-attr]
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)  # type: ignore[union-attr]
