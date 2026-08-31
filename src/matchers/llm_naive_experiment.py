"""
Phase 4, Step A - THE NAIVE APPROACH (deliberately run, not simulated).

Provider note: the project brief specified the Anthropic API
(claude-sonnet-4-6). Mid-build, three providers were tried in sequence, each
blocked at the account/billing level, not by a code defect (see
reports/llm_provider_blockers.md for the full trail of raw errors):
  1. Anthropic (claude-sonnet-4-6): 400 "credit balance too low"
  2. OpenAI (gpt-4o-mini): 429 "insufficient_quota / credit_balance_exhausted"
  3. Google Gemini: first API key's project returned 403 "denied access" on
     every current model; a second key worked.
This was ultimately built and run against the Google Gemini API
(gemini-3.5-flash-lite, GEMINI_API_KEY free tier). The tiered-pipeline
design and the naive-vs-disciplined comparison are provider-agnostic; only
the SDK call changed. This three-provider trail IS the "what broke" story
for this phase - see reports/metrics_report.md.

This script is the "what broke" experiment itself, not production code. It
answers one question honestly: what happens if we skip the tiered design
entirely and just ask an LLM to resolve everything Tier 1 couldn't?

Scope: pretend Tier 2 (fuzzy_matcher.py) doesn't exist. Take ALL 76 events
Tier 1 left unresolved - not the tiny ~22-event remainder the real pipeline
would actually hand to an LLM - and dump them into a single prompt, asking
the model to resolve everything in one shot with free-form JSON (no
response_format schema enforcement, no structured-output guarantee). This
is the fair, apples-to-apples comparison: naive-single-call vs.
tiered-pipeline, both starting from the same 76-event pool Tier 1 left
behind.

Measures, all against ground truth:
  - wall-clock latency and token cost of the single call
  - JSON parse success/failure (raw text parsing, not response_format)
  - how many of the model's claimed matches are actually correct
  - how many are hallucinated / wrong (false positives) - the number that
    matters most, since a wrong match here is exactly the audit-trail
    corruption risk Tier 1/2 were built to avoid

Run this, capture what actually happens, then compare against the real
pipeline's numbers (Tier 1: 64/140 @ 100% precision, Tier 2: 54/76 remaining
@ 100% precision -> only ~22 genuinely ambiguous events left for Tier 3's
disciplined LLM adjudication in llm_adjudicator.py).
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_matcher import load_sources, run_exact_match

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NAIVE_MODEL = "gemini-3.5-flash-lite"  # substituted for claude-sonnet-4-6 - see module docstring


def require_api_key() -> str:
    """
    Fail loudly and immediately if the API key is missing - this is a
    deliberate design choice (see the pipeline's overall AI Judgment
    story): silently proceeding without credentials, or falling back to a
    mock/simulated response, would make the "what broke" narrative
    dishonest. Load from .env (never hardcoded), then check the resolved
    environment.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "\nFATAL: GEMINI_API_KEY is not set.\n"
            "This script calls the real Google Gemini API - it will not run "
            "with a mocked or simulated response, since the whole point of "
            "Phase 4 Step A is to observe what actually happens.\n\n"
            "Fix: create a .env file in the project root containing:\n"
            "  GEMINI_API_KEY=...\n"
            "(.env is already in .gitignore - never commit the key.)\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def build_naive_prompt(unmatched_ledger: pd.DataFrame, unmatched_gateway: pd.DataFrame, unmatched_bank: pd.DataFrame) -> str:
    """
    Deliberately unstructured/naive prompting: dump all three unresolved
    tables as raw records and ask for a JSON answer via prose instruction
    only (no response_format schema enforcement). This is what a
    time-pressured first attempt at 'just use the LLM' looks like.
    """
    ledger_records = unmatched_ledger.to_dict(orient="records")
    gateway_records = unmatched_gateway.to_dict(orient="records")
    bank_records = unmatched_bank.to_dict(orient="records")

    return f"""You are reconciling three transaction data sources for a bank: a core ledger, a payment gateway settlement file, and a bank statement. Below are ALL the records that could NOT be automatically matched by simple exact-match logic. Your job is to figure out which ledger, gateway, and bank records actually correspond to the same real-world transaction, accounting for things like truncated IDs, timezone-shifted dates, split settlements (one ledger entry = multiple gateway entries), rounding differences, duplicate entries, and near-match merchant names.

CORE LEDGER records ({len(ledger_records)} rows):
{json.dumps(ledger_records, indent=2, default=str)}

GATEWAY SETTLEMENT records ({len(gateway_records)} rows):
{json.dumps(gateway_records, indent=2, default=str)}

BANK STATEMENT records ({len(bank_records)} rows):
{json.dumps(bank_records, indent=2, default=str)}

Please resolve ALL of these into matched groups. For each match you find, tell me which ledger_id, gateway_id(s), and bank_id belong together, and why. Return your answer as JSON with a list of matches, each having ledger_id, gateway_ids (list), bank_ids (list), and reasoning. Also list anything you couldn't match as unresolved."""


def run_naive_single_call(api_key: str, prompt: str):
    from google import genai

    client = genai.Client(api_key=api_key)

    start = time.monotonic()
    response = client.models.generate_content(
        model=NAIVE_MODEL,
        contents=prompt,
    )
    elapsed = time.monotonic() - start

    text = response.text or ""
    return response, text, elapsed


def try_parse_json(raw_text: str):
    """
    Naive parsing - no schema enforcement, no retry-on-failure logic. Tries
    a bare json.loads() first, then a fallback that strips markdown code
    fences (a common failure mode: the model wraps JSON in ```json ... ```
    despite being asked for "JSON" directly). Returns (parsed_or_None, method).
    """
    try:
        return json.loads(raw_text), "direct"
    except json.JSONDecodeError:
        pass

    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
        try:
            return json.loads(stripped), "stripped_code_fence"
        except json.JSONDecodeError:
            pass

    return None, "failed"


def verify_naive_matches(parsed: dict, ground_truth_path: Path = DATA_DIR / "ground_truth.csv"):
    """Score whatever matches the naive call claims against ground truth,
    same discipline as Tier 1/2's verifiers."""
    gt = pd.read_csv(ground_truth_path)
    ledger_to_event, gateway_to_event, bank_to_event = {}, {}, {}
    event_scenario = {}
    for _, row in gt.iterrows():
        event_scenario[row["event_id"]] = row["scenario"]
        if pd.notna(row["ledger_ids"]):
            for lid in str(row["ledger_ids"]).split(";"):
                ledger_to_event[lid] = row["event_id"]
        if pd.notna(row["gateway_ids"]):
            for gid in str(row["gateway_ids"]).split(";"):
                gateway_to_event[gid] = row["event_id"]
        if pd.notna(row["bank_ids"]):
            for bid in str(row["bank_ids"]).split(";"):
                bank_to_event[bid] = row["event_id"]

    matches = parsed.get("matches") if parsed else None
    if not matches:
        return {"total_claimed": 0, "correct": 0, "incorrect": 0, "precision": float("nan"),
                "incorrect_examples": [], "matched_scenario_breakdown": {}, "malformed_entries": 0}

    correct, incorrect, malformed = 0, 0, 0
    incorrect_examples = []
    matched_scenarios = []

    for m in matches:
        if not isinstance(m, dict):
            malformed += 1
            continue
        lid = m.get("ledger_id")
        gids = m.get("gateway_ids") or []
        bids = m.get("bank_ids") or []
        if isinstance(gids, str):
            gids = [gids]
        if isinstance(bids, str):
            bids = [bids]

        l_evt = ledger_to_event.get(lid)
        g_evts = {gateway_to_event.get(g) for g in gids} if gids else set()
        b_evts = {bank_to_event.get(b) for b in bids} if bids else set()

        ok = (l_evt is not None) and (g_evts == {l_evt}) and (b_evts == {l_evt})
        if ok:
            correct += 1
            matched_scenarios.append(event_scenario.get(l_evt))
        else:
            incorrect += 1
            incorrect_examples.append({
                "claimed": m, "ledger_true_event": l_evt,
                "ledger_true_scenario": event_scenario.get(l_evt),
            })

    total = correct + incorrect
    scenario_counts = pd.Series(matched_scenarios).value_counts().to_dict() if matched_scenarios else {}

    return {
        "total_claimed": total,
        "correct": correct,
        "incorrect": incorrect,
        "precision": correct / total if total else float("nan"),
        "incorrect_examples": incorrect_examples,
        "matched_scenario_breakdown": scenario_counts,
        "malformed_entries": malformed,
    }


def main():
    api_key = require_api_key()

    ledger, gateway, bank = load_sources()
    tier1 = run_exact_match(ledger, gateway, bank)

    n_ledger = len(tier1.unmatched_ledger)
    n_gateway = len(tier1.unmatched_gateway)
    n_bank = len(tier1.unmatched_bank)
    print(f"Naive Step A scope: ALL {n_ledger} ledger + {n_gateway} gateway + {n_bank} bank rows "
          f"Tier 1 left unresolved (pretending Tier 2 doesn't exist).\n")

    prompt = build_naive_prompt(tier1.unmatched_ledger, tier1.unmatched_gateway, tier1.unmatched_bank)
    prompt_chars = len(prompt)
    print(f"Prompt size: {prompt_chars:,} characters (~{prompt_chars // 4:,} tokens est.)")
    print(f"Calling {NAIVE_MODEL} with the full dump in a single request...\n")

    response, raw_text, elapsed = run_naive_single_call(api_key, prompt)

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0
    finish_reason = response.candidates[0].finish_reason if response.candidates else None

    print(f"--- Response received in {elapsed:.1f}s ---")
    print(f"Input tokens:  {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"Finish reason: {finish_reason}")

    # Gemini free-tier usage: $0 (within free-tier rate limits). Included
    # here for schema parity with the paid-provider attempts, and to make
    # cost comparisons honest if this is ever re-run against a paid tier.
    cost = 0.0
    print(f"Estimated cost: ${cost:.4f} (Gemini free tier)")

    parsed, parse_method = try_parse_json(raw_text)
    print(f"\nJSON parse: {'SUCCESS' if parsed else 'FAILED'} (method: {parse_method})")

    log_dir = DATA_DIR.parent / "reports"
    log_dir.mkdir(exist_ok=True)
    raw_log_path = log_dir / "naive_llm_raw_response.txt"
    raw_log_path.write_text(raw_text, encoding="utf-8")
    print(f"Raw response saved to: {raw_log_path}")

    if not parsed:
        print("\n!!! Could not parse any JSON from the response. This alone is a real")
        print("!!! failure mode of the naive approach - see the raw response file.")
        print(f"\nFirst 500 chars of raw response:\n{raw_text[:500]}")
        report = {"total_claimed": 0, "correct": 0, "incorrect": 0, "precision": float("nan"),
                  "malformed_entries": 0, "json_parse_failed": True}
    else:
        n_claimed = len(parsed.get("matches", []))
        print(f"Claimed matches: {n_claimed}")
        report = verify_naive_matches(parsed)
        report["json_parse_failed"] = False
        print(f"\n--- Ground truth verification ---")
        print(f"Correct: {report['correct']}/{report['total_claimed']}  "
              f"Precision: {report['precision']:.4%}")
        print(f"Malformed entries (couldn't even parse a claimed match's ids): {report['malformed_entries']}")
        print(f"Matched scenario breakdown: {report['matched_scenario_breakdown']}")
        if report["incorrect"] > 0:
            print(f"\n!!! {report['incorrect']} WRONG / HALLUCINATED MATCH(ES):")
            for ex in report["incorrect_examples"][:10]:
                print(f"  {ex}")

    summary = {
        "provider": "google_gemini",
        "model": NAIVE_MODEL,
        "scope_events": n_ledger,
        "elapsed_seconds": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 4),
        "json_parse_method": parse_method,
        **report,
    }
    summary_path = log_dir / "naive_llm_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    print("\n--- Comparison point for the metrics report ---")
    print(f"Naive single-call: resolved {report.get('correct', 0)}/{n_ledger} events correctly "
          f"({report.get('correct', 0)/n_ledger:.1%}), {report.get('incorrect', 0)} wrong matches, "
          f"in {elapsed:.1f}s for ${cost:.4f}.")
    print(f"Tiered pipeline (Tier 1+2, already measured): resolved 118/140 events (84.3%) "
          f"at 100% precision, algorithmically, before any LLM call - leaving only ~22 "
          f"genuinely ambiguous events for Tier 3's disciplined adjudication.")

    return summary


if __name__ == "__main__":
    main()
