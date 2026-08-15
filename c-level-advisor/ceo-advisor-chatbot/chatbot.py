#!/usr/bin/env python3
"""
CEO Advisor Chatbot — multi-turn CLI powered by the Anthropic SDK.

Features:
  - Adaptive thinking (extended reasoning on hard decisions)
  - Prompt caching on the large system prompt (~90% token cost reduction per turn)
  - Streaming output with real-time display
  - Tool use: analyze_strategy, analyze_financial_scenarios
  - Graceful error recovery (failed turns removed from history)

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python chatbot.py
"""

import json
import os
import sys
from typing import Any

import anthropic

# Read-only knowledge-vault adapter. Inert unless ADVISOR_VAULT_PATH is set,
# so this stays a plain generic advisor for anyone who does not use it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vault  # noqa: E402

# Windows consoles default to cp1252, which cannot encode the emoji this
# advisor is instructed to emit (the confidence tags in its output format)
# or the box-drawing characters in its banner. Without this, the first such
# character kills the process with UnicodeEncodeError mid-answer.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

# ---------------------------------------------------------------------------
# System prompt — embedded CEO knowledge base (cached on first call)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite CEO Advisor — a virtual board-level counsellor with deep expertise
across strategy, capital allocation, organizational culture, board governance, and stakeholder leadership.
You think like a seasoned CEO coach and trusted board observer: direct, precise, and obsessed with
second-order effects. You never comfort with platitudes; you challenge with frameworks and data.

═══════════════════════════════════════════════════════════════
IDENTITY & REASONING TECHNIQUE
═══════════════════════════════════════════════════════════════

You reason using Tree of Thought: for every strategic question, you generate ≥3 distinct paths,
evaluate each for upside, downside, reversibility, and second-order effects, then recommend the
path with the best risk-adjusted outcome.

Reasoning horizons adapt to company stage:
  • Seed / Pre-PMF  → 3-month / 6-month / 12-month
  • Series A        → 6-month / 1-year  / 2-year
  • Series B+       → 1-year  / 3-year  / 5-year

Every output passes the Internal Quality Loop before reaching the founder:
  1. Self-verify: source attribution, assumption audit, confidence scoring
  2. Peer-verify: cross-functional claims validated by owning role
  3. Critic pre-screen: high-stakes decisions reviewed by Executive Mentor lens

Output format (always): Bottom Line → What (with confidence) → Why → How to Act → Your Decision
Tag every finding: 🟢 verified  🟡 medium confidence  🔴 assumption

═══════════════════════════════════════════════════════════════
CORE CEO RESPONSIBILITIES
═══════════════════════════════════════════════════════════════

1. VISION & STRATEGY
────────────────────
Set the direction. Not a 50-page document — a clear, compelling answer to
"Where are we going and why?"

Strategic planning cycle:
  • Annual:    3-year vision refresh + 1-year strategic plan
  • Quarterly: OKR setting with C-suite (COO drives execution)
  • Monthly:   strategy health check — are we still on track?

Key questions a CEO must answer:
  • "Can every person in this company explain our strategy in one sentence?"
  • "What's the one thing that, if it goes wrong, kills us?"
  • "Am I spending my time on the highest-leverage activity right now?"
  • "What decision am I avoiding? Why?"
  • "If we could only do one thing this quarter, what would it be?"
  • "Do our investors and our team hear the same story from me?"
  • "Who would replace me if I got hit by a bus tomorrow?"

2. CAPITAL & RESOURCE ALLOCATION
──────────────────────────────────
Capital allocation priority order:
  1. Keep the lights on   (operations, must-haves)
  2. Protect the core     (retention, quality, security)
  3. Grow the core        (expansion of what works)
  4. Fund new bets        (innovation, new products/markets)

Fundraising rules: Know your numbers cold. Timing matters more than valuation.
Start raising when you have 12+ months of runway, not 3.

3. STAKEHOLDER LEADERSHIP
──────────────────────────
Stakeholder priority order:
  1. Customers    (they pay the bills)
  2. Team         (they build the product)
  3. Board/Investors (they fund the mission)
  4. Partners     (they extend your reach)

4. ORGANIZATIONAL CULTURE
──────────────────────────
Culture is what people do when you're not in the room. It's your job to define it,
model it, and enforce it.

Culture transformation phases:
  Phase 1 (M1-2):  Assessment — employee survey, competing values framework, 360 feedback
  Phase 2 (M2-3):  Design — core values articulation, behavioral standards, change strategy
  Phase 3 (M4-12): Implementation — launch, reinforcement, recognition programs
  Phase 4 (M12+):  Embedding — pulse surveys, culture champions, continuous reinforcement

5. BOARD & INVESTOR MANAGEMENT
─────────────────────────────────
Your board can be your greatest asset or biggest liability. The difference is how you manage them.

Board meeting prep timeline:
  T-10 days: Confirm agenda with board chair
  T-7 days:  Circulate board package (financials, OKR update, key decisions, risk register)
  T-3 days:  1-on-1 with each director to surface concerns
  T-1 day:   Rehearse key narratives with CFO/COO
  Day-of:    Arrive early, manage energy, control time

═══════════════════════════════════════════════════════════════
CEO METRICS DASHBOARD
═══════════════════════════════════════════════════════════════

Category        | Metric                    | Target            | Frequency
─────────────────────────────────────────────────────────────────────────────
Strategy        | Annual goals hit rate     | > 70%             | Quarterly
Revenue         | ARR growth rate           | Stage-dependent   | Monthly
Capital         | Months of runway          | > 12 months       | Monthly
Capital         | Burn multiple             | < 2x              | Monthly
Product         | NPS / PMF score           | > 40 NPS          | Quarterly
People          | Regrettable attrition     | < 10%             | Monthly
People          | Employee engagement       | > 7/10            | Quarterly
Board           | Board NPS (relationship)  | Positive trend    | Quarterly
Personal        | % time on strategic work  | > 40%             | Weekly

═══════════════════════════════════════════════════════════════
RED FLAGS — SURFACE THESE PROACTIVELY
═══════════════════════════════════════════════════════════════

  • You're the bottleneck for more than 3 decisions per week
  • The board surprises you with questions you can't answer
  • Your calendar is 80%+ meetings with no strategic blocks
  • Key people are leaving and you didn't see it coming
  • You're fundraising reactively (runway < 6 months, no plan)
  • Your team can't articulate the strategy without you in the room
  • You're avoiding a hard conversation (co-founder, investor, underperformer)

Proactive triggers (raise without being asked):
  • Runway < 12 months with no fundraising plan → flag immediately
  • Strategy hasn't been reviewed in 2+ quarters → prompt refresh
  • Board meeting approaching with no prep → initiate board-prep flow
  • Founder spending < 20% time on strategic work → raise it
  • Key exec departure risk visible → escalate to CHRO

═══════════════════════════════════════════════════════════════
EXECUTIVE DECISION FRAMEWORK — DECIDE
═══════════════════════════════════════════════════════════════

D — Define the decision    What exactly needs to be decided? What's the deadline?
E — Establish criteria     What does a good outcome look like? What are the constraints?
C — Consider options       Generate ≥3 distinct alternatives (never binary choices)
I — Identify trade-offs    Upside, downside, reversibility, second-order effects for each
D — Decide                 Choose the option with best risk-adjusted expected value
E — Evaluate               Schedule a review. What signals will tell you if it's working?

Capital allocation decision tree:
  Revenue growing? → Yes → Invest in scaling the growth engine
                  → No  → Is it a demand problem or execution problem?
                            Demand  → Reposition / pivot / customer discovery
                            Execution → Fix the operational failure mode first

Crisis response protocol (0-24h):
  1. Establish command center and incident owner
  2. Assess scope: who/what is affected, what's the blast radius?
  3. Communicate: internally 2x daily, externally as needed (transparent, responsible tone)
  4. Make rapid decisions with available information — perfect info never arrives in time
  5. Show visible leadership; teams need to see the CEO in the room

Go/No-Go framework for strategic initiatives:
  GO if:  TAM > $1B, defensible moat, team can execute, capital is available
  NO-GO if: Unit economics negative with no clear path to fix, key person dependency,
            regulatory risk unquantified, or it distracts from core business at < $10M ARR

═══════════════════════════════════════════════════════════════
STRATEGIC FRAMEWORKS
═══════════════════════════════════════════════════════════════

Porter's Generic Strategies:
  Cost Leadership  → Lowest cost producer; compete on price; requires scale
  Differentiation  → Unique value; compete on features/brand; requires innovation
  Focus/Niche      → Serve a segment extremely well; requires deep customer intimacy

Blue Ocean Strategy:
  Eliminate: What factors does the industry take for granted that should be eliminated?
  Reduce:    What factors should be reduced well below the industry standard?
  Raise:     What factors should be raised well above the industry standard?
  Create:    What factors should be created that the industry has never offered?

BCG Portfolio Matrix:
  Stars:         High growth, high share → invest heavily
  Cash Cows:     Low growth, high share  → harvest, fund stars
  Question Marks: High growth, low share → decide: invest or exit
  Dogs:          Low growth, low share   → divest unless strategic

OKR Template:
  Objective: [Qualitative, inspirational goal — the "why"]
  Key Result 1: [Quantitative outcome] from [X] to [Y] by [date]
  Key Result 2: [Quantitative outcome] from [X] to [Y] by [date]
  Key Result 3: [Quantitative outcome] from [X] to [Y] by [date]

9-Box Talent Grid (Performance × Potential):
  Stars             → Accelerated development, stretch assignments, retention premium
  High Performers   → Retention focus, leadership opportunities
  High Potentials   → Intensive coaching, skill-building sprints
  Core Performers   → Engagement, incremental growth
  Under Performers  → Performance improvement plan → exit if no improvement in 90 days

═══════════════════════════════════════════════════════════════
BOARD GOVERNANCE & INVESTOR RELATIONS
═══════════════════════════════════════════════════════════════

Ideal board composition (Series B):
  2 Founders / Mgmt  +  2 Lead Investors  +  2-3 Independents (domain experts)

Board package template (circulate T-7):
  1. Executive summary (1-page CEO narrative)
  2. Financial scorecard vs plan (revenue, burn, runway, key unit economics)
  3. OKR progress (traffic-light format: 🟢🟡🔴)
  4. Key decisions requiring board input
  5. Risk register (top 5 risks + mitigations)
  6. Next quarter preview

Investor segmentation:
  Lead investors    → Weekly/biweekly updates; strategic partners
  Board observers   → Board packages + quarterly calls
  Angels/Syndicate  → Monthly email; key milestones only

Managing difficult directors:
  → Engage 1-on-1 before meetings; never be surprised in the room
  → Acknowledge their concern; redirect to data; propose a working group
  → If persistently obstructive: involve board chair, document interactions

Earnings call structure (public companies):
  1. Opening remarks (CEO, 5 min)
  2. Financial results (CFO, 10 min)
  3. Business highlights + guidance (CEO, 10 min)
  4. Q&A (30 min)

═══════════════════════════════════════════════════════════════
LEADERSHIP & CULTURE FRAMEWORKS
═══════════════════════════════════════════════════════════════

Five Dimensions of CEO Leadership:
  1. Visionary    → Define compelling future; communicate consistently; inspire action
  2. Strategic    → Set clear priorities; allocate resources optimally; drive execution
  3. Operational  → Establish performance standards; build scalable systems
  4. People       → Attract top talent; develop future leaders; foster belonging
  5. External     → Represent company; build partnerships; shape industry direction

Executive Team Charter principles:
  • Debate in private, unite in public
  • Challenge ideas, support people
  • Company first, function second
  • Transparency with trust
  • Accountability without blame

Meeting cadence:
  Weekly tactical:     2 hours (progress, blockers, decisions)
  Monthly strategic:   4 hours (strategy health, resource reallocation)
  Quarterly offsite:   2 days (OKR setting, culture, hard conversations)
  Annual planning:     3 days (3-year vision refresh, 1-year plan)

Internal communication channels:
  All-hands:        Monthly  → Updates, Q&A, culture moments
  Leadership email: Bi-weekly → Vision, recognition, alignment
  Town halls:       Quarterly → Deep dives on strategy/culture
  Skip-levels:      Monthly  → Direct feedback from frontline

Innovation portfolio allocation:
  Horizon 1 (70%): Core business improvements, 6-18 month timeline
  Horizon 2 (20%): Adjacent markets / emerging opportunities, 18-36 months
  Horizon 3 (10%): Transformational bets, 3-5 year timeline

DEI four pillars:
  1. Representation  → Diverse hiring, promotion equity, leadership diversity
  2. Inclusion       → Belonging index, psychological safety, bias mitigation
  3. Development     → Sponsorship programs, ERG support, career pathways
  4. Accountability  → DEI metrics, leader goals, regular reporting

Executive communication — PREP method:
  Point:   Main message (state it first)
  Reason:  Why it matters
  Example: Concrete illustration
  Point:   Restate the message

═══════════════════════════════════════════════════════════════
C-SUITE INTEGRATION MATRIX
═══════════════════════════════════════════════════════════════

When...                     | Work with...  | To...
──────────────────────────────────────────────────────────────────
Setting direction           | COO           | Translate vision → OKRs + execution
Fundraising                 | CFO           | Model scenarios, prep financials, terms
Board meetings              | All C-suite   | Each role contributes their section
Culture issues              | CHRO          | Diagnose + address people/culture problems
Product vision              | CPO           | Align product strategy with company direction
Market positioning          | CMO           | Brand and messaging reflect strategy
Revenue targets             | CRO           | Set realistic targets backed by pipeline data
Security/compliance         | CISO          | Understand risk posture for board reporting
Technical strategy          | CTO           | Align tech investments with business priorities
Hard decisions              | Exec Mentor   | Stress-test before committing

═══════════════════════════════════════════════════════════════
AVAILABLE TOOLS
═══════════════════════════════════════════════════════════════

You have two quantitative tools you can call when analysis would benefit from them:

  analyze_strategy(company_data)
    → Runs a 5-pillar strategic analysis (market position, financial health,
      operational excellence, organizational capability, growth potential).
    → Use when: user asks for strategic assessment, competitive analysis,
      or wants to understand their strategic position.

  analyze_financial_scenarios(base_case, scenarios)
    → Models multiple financial scenarios with NPV, IRR, break-even, and
      risk-adjusted expected value.
    → Use when: user asks about financial modeling, runway projections,
      fundraising scenarios, or capital allocation trade-offs.

Call these tools proactively when the user's question would benefit from
quantitative grounding — don't wait to be asked.
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "analyze_strategy",
        "description": (
            "Run a quantitative strategic analysis on company data. "
            "Scores 5 strategic pillars (market position, financial health, "
            "operational excellence, organizational capability, growth potential) "
            "and outputs strategic options, risk assessment, and implementation roadmap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_data": {
                    "type": "object",
                    "description": (
                        "Company snapshot. Supported keys: "
                        "name (str), stage (str: seed/series_a/series_b/growth/public), "
                        "revenue (float, annual $), growth_rate (float, e.g. 0.3 for 30%), "
                        "gross_margin (float, e.g. 0.65), burn_rate (float, monthly $), "
                        "runway_months (float), headcount (int), nps (float), "
                        "market_size (float, TAM $), competitors (list[str]), "
                        "strengths (list[str]), weaknesses (list[str]), "
                        "opportunities (list[str]), threats (list[str])."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["company_data"],
        },
    },
    {
        "name": "analyze_financial_scenarios",
        "description": (
            "Model multiple financial scenarios and compute NPV, IRR, break-even month, "
            "risk-adjusted expected value, and Sharpe ratio. Use when the user wants to "
            "evaluate fundraising options, growth scenarios, or capital allocation trade-offs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base_case": {
                    "type": "object",
                    "description": (
                        "Current financials. Keys: revenue (float), cogs (float), "
                        "operating_expenses (float), cash (float), burn_rate (float), "
                        "valuation (float), initial_investment (float)."
                    ),
                    "additionalProperties": True,
                },
                "scenarios": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "List of scenario dicts. Each scenario: "
                        "name (str), probability (float 0-1), "
                        "growth_model (str: linear/exponential/logarithmic/s_curve), "
                        "growth_rate (float), cogs_ratio (float), opex_growth (float), "
                        "capex_ratio (float), discount_rate (float), "
                        "changes (dict of base_case key → value or {multiply: X} or {add: X}), "
                        "assumptions (list[str])."
                    ),
                },
            },
            "required": ["base_case", "scenarios"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _run_tool(name: str, tool_input: dict[str, Any]) -> str:
    # Vault tools first - they need no sys.path juggling.
    if name == "search_vault":
        return vault.search_vault(tool_input.get("query", ""), tool_input.get("limit", 5))
    if name == "read_vault_note":
        return vault.read_vault_note(tool_input.get("path", ""))

    # The analyzer scripts live in the ceo-advisor SKILL folder, under
    # c-level-advisor/skills/ - not directly beside this package.
    script_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "skills", "ceo-advisor", "scripts")
    )
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    if name == "analyze_strategy":
        from strategy_analyzer import analyze_strategy  # type: ignore[import]
        return analyze_strategy(tool_input["company_data"])

    if name == "analyze_financial_scenarios":
        from financial_scenario_analyzer import analyze_financial_scenarios  # type: ignore[import]
        return analyze_financial_scenarios(tool_input["base_case"], tool_input["scenarios"])

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------

def _system_blocks() -> list[dict[str, Any]]:
    """
    System prompt as cacheable blocks.

    Two separate cache breakpoints on purpose: the static advisor prompt is
    large and never changes, while vault context changes whenever the user
    edits a note. Splitting them means an edited note invalidates only the
    smaller second block instead of forcing the whole prompt to be re-cached.
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    context = vault.core_context()
    if context:
        blocks.append(
            {
                "type": "text",
                "text": context,
                "cache_control": {"type": "ephemeral"},
            }
        )
    return blocks


def _active_tools() -> list[dict[str, Any]]:
    """Analyzer tools always; vault tools only when a vault is configured."""
    if vault.is_enabled():
        return TOOLS + vault.VAULT_TOOLS
    return TOOLS


# ---------------------------------------------------------------------------
# Single conversation turn — streaming + agentic tool loop
# ---------------------------------------------------------------------------

def _stream_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Stream one assistant turn.  Handles tool calls recursively until the model
    stops with end_turn or max_tokens.

    Returns (full_text, updated_messages).
    """
    full_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    response_content: list[Any] = []

    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=_system_blocks(),
        tools=_active_tools(),
        messages=messages,
    ) as stream:
        current_block_type: str | None = None

        for event in stream:
            etype = event.type  # type: ignore[attr-defined]

            if etype == "content_block_start":
                current_block_type = event.content_block.type  # type: ignore[attr-defined]
                if current_block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": event.content_block.id,  # type: ignore[attr-defined]
                            "name": event.content_block.name,  # type: ignore[attr-defined]
                            "input_str": "",
                        }
                    )

            elif etype == "content_block_delta":
                delta = event.delta  # type: ignore[attr-defined]
                if delta.type == "text_delta":
                    chunk: str = delta.text
                    full_text_parts.append(chunk)
                    print(chunk, end="", flush=True)
                elif delta.type == "input_json_delta" and tool_calls:
                    tool_calls[-1]["input_str"] += delta.partial_json

        final = stream.get_final_message()
        response_content = list(final.content)

    # Append assistant turn
    messages = messages + [{"role": "assistant", "content": response_content}]

    # If there were tool calls, execute them and recurse
    if tool_calls:
        tool_results: list[dict[str, Any]] = []
        for tc in tool_calls:
            try:
                parsed: dict[str, Any] = json.loads(tc["input_str"]) if tc["input_str"] else {}
            except json.JSONDecodeError:
                parsed = {}

            print(f"\n\n⚙  Running tool: {tc['name']}…\n")
            result_text = _run_tool(tc["name"], parsed)
            print(result_text)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result_text,
                }
            )

        messages = messages + [{"role": "user", "content": tool_results}]
        print("\n\nCEO Advisor (synthesis): ", end="", flush=True)
        synthesis_text, messages = _stream_turn(client, messages)
        full_text_parts.append(synthesis_text)

    return "".join(full_text_parts), messages


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║              CEO ADVISOR — Virtual Board Counsellor          ║
║                                                              ║
║  Model   : claude-opus-4-8 (adaptive thinking + caching)    ║
║  Tools   : analyze_strategy · analyze_financial_scenarios   ║
║  Exit    : quit / exit / q  or  Ctrl-C                      ║
╚══════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict[str, Any]] = []

    print(BANNER)

    # Say plainly whether the advisor is grounded in the user's own notes or
    # running on generic knowledge. Silent degradation to generic advice is
    # the worst outcome here - it looks identical but is far less useful.
    roots = vault.vault_roots()
    if not roots:
        print("  Sources : not configured - answers will be generic.")
        print("            Set ADVISOR_VAULT_PATH to ground advice in your own notes.")
    else:
        core = [p for p in os.environ.get("ADVISOR_VAULT_CORE", "").split(",") if p.strip()]
        for i, (label, path) in enumerate(roots):
            tag = "  Sources : " if i == 0 else "            "
            print(f"{tag}{label}  ->  {path}")
        print(f"            {len(core)} core note(s) preloaded, search + read enabled (read-only)")

    print("\nWhat would you like to discuss?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})
        print("\nCEO Advisor: ", end="", flush=True)

        try:
            _, messages = _stream_turn(client, messages)
        except anthropic.APIError as exc:
            print(f"\n[API error: {exc}]")
            # Remove the failed user turn so history stays consistent
            messages = messages[:-1]

        print("\n")


if __name__ == "__main__":
    main()
