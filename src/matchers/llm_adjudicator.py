"""
Tier 3 - LLM adjudication (Phase 4, Step B: the disciplined rebuild).

Provider note: the brief originally specified the Anthropic API
(claude-sonnet-4-6). Getting there required three provider attempts -
Anthropic (no usable credits), OpenAI (no usable credits), and Google
Gemini (first key blocked at the project level, second key worked) - see
reports/llm_provider_blockers.md for the full trail of raw errors. Google
Gemini (gemini-3.5-flash-lite, GEMINI_API_KEY, free tier) is the model this
project runs Tier 3 against. The tiered design and adjudication methodology
are provider-agnostic: swapping the model back to claude-sonnet-4-6 is a
model-name and SDK-call change, not a redesign - see
adjudicate_one()/RESPONSE_SCHEMA/SYSTEM_INSTRUCTION below, which carry
directly. The free-tier rate limit (15 req/min, see
FREE_TIER_REQUESTS_PER_MINUTE) is an accepted, documented constraint of
this choice, not an oversight - see reports/metrics_report.md for the
throughput tradeoff this implies.

WHAT THIS TIER IS FOR (and what it deliberately is NOT for)
-------------------------------------------------------------
Tier 1 (exact_matcher.py) and Tier 2 (fuzzy_matcher.py) already resolved
118/140 events (84.3%) algorithmically, at 100% measured precision, before
any LLM call. What's left after both tiers - roughly 22 ledger rows in this
dataset - are the events that are genuinely, structurally ambiguous:
  - duplicate: two rows on one side are raw-identical (same id, amount,
    date) - is this a true duplicate write, or two separate legitimate
    payments that coincidentally match? A rule can't tell; this requires
    judgment about plausibility (would this merchant realistically get
    paid twice, same amount, same day?).
  - orphan: a ledger row has no true counterpart in one other source. This
    is not a matching problem for the LLM to "solve" by finding a partial
    or approximate match - the correct behavior is recognizing there is
    nothing to match and saying so.

Step A (llm_naive_experiment.py) already showed what happens without this
discipline: given the SAME starting pool (all 76 Tier-1 leftovers, i.e.
Tier 2 pretended not to exist), a single naive LLM call achieved only
82.7% precision - 13 wrong matches, 9 of which were orphan events the model
confidently FORCE-MATCHED to a plausible-looking but false counterpart,
inventing a rationale ("matched by transaction ID, amount, and merchant")
that didn't hold up against ground truth. That is exactly the audit-trail
corruption risk this tier exists to avoid.

Design choices that directly target that failure mode:
  1. Structured JSON output (response_schema) - not prompt-and-hope. The
     schema makes "no_match" a first-class, always-available verdict, not
     something the model has to volunteer against the grain of a "resolve
     everything" framing.
  2. The prompt explicitly tells the model that "no true match exists" is
     a CORRECT and EXPECTED answer for some records, and that guessing is
     worse than abstaining - directly countering the failure mode observed
     in Step A.
  3. One LLM call per ledger row (not one giant dump) - keeps context small
     and focused, and means a bad call on one row can't contaminate
     judgment on another the way a single 76-row dump does.
  4. Every decision - match, no-match, or uncertain - is logged with a
     rationale and confidence via audit_log.py (Phase 5), regardless of
     outcome.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_matcher import load_sources, run_exact_match
from fuzzy_matcher import run_tier2

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ADJUDICATOR_MODEL = "gemini-3.5-flash-lite"  # substituted for claude-sonnet-4-6 - see module docstring

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["match", "no_match", "uncertain"],
            "description": "match = you found a confident counterpart on AT LEAST ONE of gateway/bank "
                            "(set that id; leave the OTHER id null if no true counterpart exists on that "
                            "side - a one-sided match is a complete, correct answer, not a partial "
                            "failure - do not withhold a confident single-side match just because the "
                            "other side has nothing). no_match = there is genuinely no confident "
                            "counterpart on EITHER side (correct for a duplicate row that is plausibly "
                            "two separate real payments, or if truly nothing plausible exists anywhere). "
                            "uncertain = insufficient evidence either way; do not guess.",
        },
        "matched_gateway_id": {
            "type": "string",
            "nullable": True,
            "description": "gateway_id of the matched row, or null if there is no true gateway "
                            "counterpart for this ledger row (this is a normal, valid outcome, not a "
                            "failure - see is_partial_match). Only set when verdict is 'match'.",
        },
        "matched_bank_id": {
            "type": "string",
            "nullable": True,
            "description": "bank_id of the matched row, or null if there is no true bank counterpart "
                            "for this ledger row (normal, valid outcome). Only set when verdict is 'match'.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0. How confident the verdict is, independent of what the verdict is - "
                            "a confident no_match should still score high confidence.",
        },
        "rationale": {
            "type": "string",
            "description": "One or two sentences explaining the verdict, referencing the specific "
                            "evidence considered (amounts, dates, merchant names, ids). Must be honest "
                            "about uncertainty - do not state something is 'exact' or 'confirmed' unless "
                            "it genuinely is.",
        },
    },
    "required": ["verdict", "matched_gateway_id", "matched_bank_id", "confidence", "rationale"],
}

SYSTEM_INSTRUCTION = """You are adjudicating the small set of transactions a two-stage automated \
reconciliation pipeline (exact-match, then fuzzy/algorithmic matching) could NOT resolve. Every \
record you see here already survived both of those stages - if a confident deterministic or \
tolerance-based match existed, it would have been made already. That means what remains is \
genuinely ambiguous, and a meaningful fraction of these ledger rows have NO true match in one of \
the other sources at all (they are correctly unresolved, not solvable).

Your job is NOT to force a match rate as high as possible. Your job is to be right. Concretely:
- A ONE-SIDED match (a confident gateway match with no bank counterpart, or vice versa) is a \
COMPLETE and CORRECT answer, not a partial failure. This dataset genuinely contains records that \
are missing from exactly one of the two other sources (e.g. a gateway settlement that never \
appeared on the bank statement yet, or a bank entry with no corresponding gateway row). Do NOT \
withhold a confident match on one side just because the other side is empty - if you're confident \
about gateway but there is truly no plausible bank candidate, return verdict: match with \
matched_gateway_id set and matched_bank_id null. Do not describe this as "cannot be fully matched" \
or wait for "three-way reconciliation" - a two-way match IS the correct final answer here, there is \
no further stage that will add the missing side.
- Only return verdict: no_match when NEITHER side has a plausible candidate at all.
- If two candidate rows on the SAME side are equally plausible and you cannot genuinely distinguish \
them, say uncertain rather than picking one arbitrarily.
- Only claim a side as matched when you have real, specific evidence (not just "the amount is \
similar") that a particular gateway_id or bank_id is the same transaction as the ledger row.
- Minor timestamp differences (a bank value_datetime a few hours before or after the ledger \
txn_datetime, even crossing midnight) are NORMAL settlement/processing lag, not evidence against a \
match - do not treat time-of-day ordering as disqualifying when id, amount, and merchant already \
agree. Only the calendar date being wildly different, or amount/id actually disagreeing, is real \
evidence against a match.
- A wrong match is worse than a missed one. A missed one falls through to a human exception queue \
un-harmfully; a wrong match corrupts the reconciliation audit trail. When genuinely torn between \
match and no_match/uncertain, choose the more conservative option - but "torn" means real \
conflicting evidence, not the mere absence of a candidate on the other side.
- For "duplicate"-shaped situations (two candidate rows that are near-identical to each other in \
amount/date), consider whether it's plausible for this merchant to legitimately receive two \
separate payments of the same amount on the same day (e.g. two similar-sized retail transactions) \
versus a system double-write. State your reasoning either way - don't just guess."""


@dataclass
class AdjudicationResult:
    ledger_id: str
    verdict: str  # match | no_match | uncertain
    matched_gateway_id: str = None
    matched_bank_id: str = None
    confidence: float = 0.0
    rationale: str = ""
    tier: str = "tier3_llm"
    candidates_considered: dict = field(default_factory=dict)


def require_api_key() -> str:
    """Fail loudly if the key is missing - see llm_naive_experiment.py for
    the same discipline and why it matters here."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "\nFATAL: GEMINI_API_KEY is not set.\n"
            "Tier 3 calls the real Google Gemini API - it will not run with "
            "a mocked or simulated response.\n\n"
            "Fix: create a .env file in the project root containing:\n"
            "  GEMINI_API_KEY=...\n"
            "(.env is already in .gitignore - never commit the key.)\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def build_candidate_pool(ledger_row, gateway_pool: pd.DataFrame, bank_pool: pd.DataFrame, amount_window: float = 500.0):
    """
    For one ledger row, narrow the gateway/bank pool to plausible
    candidates before calling the LLM - a wide amount window (not a tight
    tolerance; Tier 2 already tried tight tolerances) so the model has
    real alternatives to reason about, including the "these are actually
    different transactions" case, without paying for the entire remaining
    pool on every call.
    """
    l_amount = round(float(ledger_row["amount"]), 2)
    gw_candidates = gateway_pool[
        (gateway_pool["amount"] - l_amount).abs() <= amount_window
    ].to_dict(orient="records")
    bk_candidates = bank_pool[
        (bank_pool["amount"] - l_amount).abs() <= amount_window
    ].to_dict(orient="records")
    return gw_candidates, bk_candidates


def adjudicate_one(client, ledger_row, gw_candidates, bk_candidates) -> AdjudicationResult:
    from google.genai import types

    prompt = f"""LEDGER RECORD (needs a decision):
{json.dumps(ledger_row, indent=2, default=str)}

CANDIDATE GATEWAY SETTLEMENT RECORDS ({len(gw_candidates)} in range):
{json.dumps(gw_candidates, indent=2, default=str)}

CANDIDATE BANK STATEMENT RECORDS ({len(bk_candidates)} in range):
{json.dumps(bk_candidates, indent=2, default=str)}

Adjudicate this ledger record: is there a true match among the candidates, or not?"""

    response = client.models.generate_content(
        model=ADJUDICATOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )

    parsed = json.loads(response.text)
    return AdjudicationResult(
        ledger_id=ledger_row["ledger_id"],
        verdict=parsed["verdict"],
        matched_gateway_id=parsed.get("matched_gateway_id"),
        matched_bank_id=parsed.get("matched_bank_id"),
        confidence=float(parsed.get("confidence", 0.0)),
        rationale=parsed.get("rationale", ""),
        candidates_considered={
            "gateway_ids": [c["gateway_id"] for c in gw_candidates],
            "bank_ids": [c["bank_id"] for c in bk_candidates],
        },
    )


FREE_TIER_REQUESTS_PER_MINUTE = 15  # gemini-3.5-flash-lite free-tier limit, observed via a live 429
SECONDS_PER_REQUEST = 60.0 / FREE_TIER_REQUESTS_PER_MINUTE
MAX_RETRIES_PER_ROW = 3


def _call_with_retry(fn, *args, **kwargs):
    """Retry on 429 (rate limit) with the delay the API itself reports,
    plus a small buffer - the free tier's 15 req/min limit is a hard
    external constraint, not a bug, so backing off and retrying is the
    correct behavior rather than failing the whole run."""
    import re
    from google.genai import errors as genai_errors

    for attempt in range(MAX_RETRIES_PER_ROW):
        try:
            return fn(*args, **kwargs)
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e) or attempt == MAX_RETRIES_PER_ROW - 1:
                raise
            match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", str(e))
            delay = int(match.group(1)) + 2 if match else 15
            print(f"  Rate limited, retrying in {delay}s (attempt {attempt+1}/{MAX_RETRIES_PER_ROW})...")
            time.sleep(delay)


def run_tier3(unmatched_ledger: pd.DataFrame, unmatched_gateway: pd.DataFrame, unmatched_bank: pd.DataFrame, api_key: str):
    from google import genai

    client = genai.Client(api_key=api_key)

    results = []
    for i, (_, lrow) in enumerate(unmatched_ledger.iterrows()):
        ledger_dict = lrow.to_dict()
        gw_candidates, bk_candidates = build_candidate_pool(ledger_dict, unmatched_gateway, unmatched_bank)
        result = _call_with_retry(adjudicate_one, client, ledger_dict, gw_candidates, bk_candidates)
        results.append(result)
        print(f"  [{i+1}/{len(unmatched_ledger)}] {result.ledger_id}: {result.verdict}")
        if i < len(unmatched_ledger) - 1:
            time.sleep(SECONDS_PER_REQUEST)  # stay under the free-tier 15 req/min limit
    return results


def verify_against_ground_truth(results: list, ground_truth_path: Path = DATA_DIR / "ground_truth.csv"):
    """
    Same discipline as Tier 1/2's verifiers, adapted for a three-way
    verdict (match/no_match/uncertain) rather than a binary matched/not:
      - match precision: of the rows where Tier 3 claimed a match, how
        many are actually correct
      - no_match / uncertain correctness: for orphan and duplicate events,
        was declining to match (or flagging uncertain) the RIGHT call, or
        did Tier 3 miss a real match that was actually resolvable?
    """
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

    match_correct, match_incorrect = 0, 0
    match_incorrect_examples = []
    no_match_correct, no_match_incorrect = 0, 0
    uncertain_count = 0
    by_scenario = {}

    for r in results:
        l_evt = ledger_to_event.get(r.ledger_id)
        scenario = event_scenario.get(l_evt)
        by_scenario.setdefault(scenario, {"match": 0, "no_match": 0, "uncertain": 0})
        by_scenario[scenario][r.verdict] += 1

        # Ground truth per side: does this ledger row have a TRUE gateway
        # counterpart, and a TRUE bank counterpart? For most scenarios both
        # are true (a full three-way match exists). For `orphan` events,
        # by construction, exactly one side is missing - the correct
        # adjudication is a PARTIAL match (the side that exists) with the
        # other side left null, not a three-way match and not a blanket
        # no_match. `duplicate` events have a true counterpart on both
        # sides too (one of the two candidate rows on the ambiguous side is
        # the real one) - but which specific row is genuinely ambiguous by
        # design, so ANY claimed id on that side, right or wrong, doesn't
        # by itself prove the model wasn't guessing; scored the same way
        # (does the claimed id's event match) since we have no better
        # ground-truth signal to distinguish "correct reasoning" from
        # "lucky guess" here.
        gt_row = gt[gt["event_id"] == l_evt].iloc[0] if l_evt else None
        has_true_gateway = bool(gt_row is not None and pd.notna(gt_row["gateway_ids"]) and gt_row["gateway_ids"])
        has_true_bank = bool(gt_row is not None and pd.notna(gt_row["bank_ids"]) and gt_row["bank_ids"])
        truly_resolvable = has_true_gateway or has_true_bank  # at least one true side exists

        # A side is "tied" when ground truth itself lists more than one id
        # for that side of this event (the duplicate scenario's shape) -
        # there is no single correct id to pick, so both guessing one of
        # the tied ids AND honestly abstaining (null) on that side count as
        # correct; only a claim resolving to a DIFFERENT event is wrong.
        gateway_tied = has_true_gateway and ";" in str(gt_row["gateway_ids"])
        bank_tied = has_true_bank and ";" in str(gt_row["bank_ids"])

        if r.verdict == "match":
            g_evt = gateway_to_event.get(r.matched_gateway_id) if r.matched_gateway_id else None
            b_evt = bank_to_event.get(r.matched_bank_id) if r.matched_bank_id else None

            # A claimed side is correct if: (a) the ledger row has no true
            # counterpart on that side and the model correctly left it
            # null, (b) the side is tied (duplicate-shaped) and the model
            # either abstained (null) or picked one of the tied ids, or
            # (c) the ledger row has exactly one true counterpart and the
            # model's claimed id resolves to that same event.
            gateway_side_ok = (
                (not has_true_gateway and r.matched_gateway_id is None)
                or (gateway_tied and (r.matched_gateway_id is None or g_evt == l_evt))
                or (has_true_gateway and not gateway_tied and g_evt == l_evt)
            )
            bank_side_ok = (
                (not has_true_bank and r.matched_bank_id is None)
                or (bank_tied and (r.matched_bank_id is None or b_evt == l_evt))
                or (has_true_bank and not bank_tied and b_evt == l_evt)
            )
            # At least one side must actually be claimed (a "match" verdict
            # with both ids null is not really a match) and every claimed
            # side must be correct.
            claimed_something = r.matched_gateway_id is not None or r.matched_bank_id is not None
            ok = claimed_something and gateway_side_ok and bank_side_ok

            if ok:
                match_correct += 1
            else:
                match_incorrect += 1
                match_incorrect_examples.append({
                    "ledger_id": r.ledger_id, "claimed_gateway": r.matched_gateway_id,
                    "claimed_bank": r.matched_bank_id, "true_event": l_evt,
                    "true_scenario": scenario, "rationale": r.rationale,
                })
        elif r.verdict == "no_match":
            if not truly_resolvable:
                no_match_correct += 1
            else:
                no_match_incorrect += 1  # a real match existed on at least one side and Tier 3 missed it entirely
        else:  # uncertain
            uncertain_count += 1

    total = len(results)
    return {
        "total_adjudicated": total,
        "match_verdicts": match_correct + match_incorrect,
        "match_correct": match_correct,
        "match_incorrect": match_incorrect,
        "match_precision": match_correct / (match_correct + match_incorrect) if (match_correct + match_incorrect) else float("nan"),
        "match_incorrect_examples": match_incorrect_examples,
        "no_match_verdicts": no_match_correct + no_match_incorrect,
        "no_match_correct": no_match_correct,
        "no_match_incorrect_missed_real_match": no_match_incorrect,
        "uncertain_verdicts": uncertain_count,
        "breakdown_by_scenario": by_scenario,
    }


def main():
    api_key = require_api_key()

    ledger, gateway, bank = load_sources()
    tier1 = run_exact_match(ledger, gateway, bank)
    combined, split_out, fuzzy_out = run_tier2(
        tier1.unmatched_ledger, tier1.unmatched_gateway, tier1.unmatched_bank
    )

    n_ledger = len(combined.unmatched_ledger)
    print(f"Tier 3 scope: {n_ledger} ledger rows remain after Tier 1 ({len(tier1.matches)} resolved) "
          f"and Tier 2 ({len(combined.matches)} resolved).\n")
    print(f"Adjudicating each of the {n_ledger} rows individually via {ADJUDICATOR_MODEL} "
          f"with structured JSON output...\n")

    results = run_tier3(combined.unmatched_ledger, combined.unmatched_gateway, combined.unmatched_bank, api_key)

    verdict_counts = pd.Series([r.verdict for r in results]).value_counts().to_dict()
    print(f"Verdicts: {verdict_counts}")

    report = verify_against_ground_truth(results)
    print()
    print("--- Ground truth verification (Tier 3) ---")
    print(f"Match verdicts: {report['match_verdicts']}  Correct: {report['match_correct']}  "
          f"Precision: {report['match_precision']:.4%}")
    print(f"No-match verdicts: {report['no_match_verdicts']}  "
          f"Correctly-abstained: {report['no_match_correct']}  "
          f"Missed-a-real-match: {report['no_match_incorrect_missed_real_match']}")
    print(f"Uncertain verdicts: {report['uncertain_verdicts']}")
    print(f"Breakdown by scenario: {json.dumps(report['breakdown_by_scenario'], indent=2)}")

    if report["match_incorrect"] > 0:
        print(f"\n!!! {report['match_incorrect']} WRONG MATCH VERDICT(S):")
        for ex in report["match_incorrect_examples"]:
            print(f"  {ex}")

    log_dir = DATA_DIR.parent / "reports"
    log_dir.mkdir(exist_ok=True)
    results_path = log_dir / "tier3_adjudication_results.json"
    results_path.write_text(
        json.dumps([vars(r) for r in results], indent=2, default=str), encoding="utf-8"
    )
    summary_path = log_dir / "tier3_adjudication_summary.json"
    summary_path.write_text(json.dumps({"provider": "google_gemini", "model": ADJUDICATOR_MODEL, **report}, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    return results, report


if __name__ == "__main__":
    main()
