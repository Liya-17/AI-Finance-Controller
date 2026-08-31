"""
Tier 2 - fuzzy/algorithmic matcher.

Operates only on what Tier 1 (exact_matcher.py) left unmatched. Resolves four
distinct failure modes via deterministic-but-not-raw-exact logic, plus one
separate subset-sum routine:

  id_truncated      -> normalize both sides' ID (strip "TXN" prefix,
                        case-fold, strip non-alphanumerics) then compare.
  timezone_shift     -> date-window tolerance (+/-1 calendar day), with ID
                        and amount still required exact.
  rounding_drift     -> amount-tolerance band (sub-rupee, < Re 1), with ID
                        and date still required exact.
  near_match_name    -> normalized ID (it carries a reformatted txn_id, see
                        data/generate_synthetic_data.py) + fuzzy name
                        similarity (rapidfuzz) + amount/date agreement,
                        combined - name similarity alone is never sufficient.
  split_settlement   -> handled SEPARATELY by subset-sum grouping: find a
                        set of gateway rows (same merchant + date window)
                        whose amounts sum to one ledger row's amount. This is
                        NOT routed through the pairwise fuzzy scorer above -
                        it is a different shape of problem (one-to-many) and
                        forcing it through a 1:1 scorer would silently miss
                        it or produce partial/wrong groupings.

PRECISION OVER RECALL (deliberate design choice)
--------------------------------------------------
Every Tier 2 match is treated as settled truth by the audit trail, same as
Tier 1. A wrong fuzzy match is worse than an unresolved one: a miss falls
through honestly to Tier 3 (LLM adjudication) or the exceptions bucket,
while a wrong match corrupts everything built on top of it. Where a
confidence score sits near the acceptance threshold, this module is tuned to
NOT match rather than guess - see CONFIDENCE_THRESHOLD and
SPLIT_SETTLEMENT_MIN_CONFIDENCE below. This deliberately trades recall for
precision; the tradeoff is measured and reported by
verify_against_ground_truth(), not just asserted.

Two things this tier must never do (mirrors the Tier 1 discipline):
  - Match a `duplicate`-scenario event. Two rows with identical/near-identical
    amount+date are exactly what this tier's tolerance logic is built to
    catch - which is precisely why duplicates are dangerous here. A
    duplicate is genuinely ambiguous (true duplicate vs. two legitimate
    separate payments) and is reserved for Tier 3's judgment, not a rule.
  - Match an `orphan`-scenario event. It has no true counterpart in one
    source by construction; nothing should be force-matched to fill the gap.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# --- Tunable thresholds (precision-over-recall: conservative on purpose) ---
DATE_WINDOW_DAYS = 1                 # timezone_shift tolerance
AMOUNT_TOLERANCE_RUPEES = 0.99       # rounding_drift tolerance (sub-rupee only)
NAME_FUZZY_MIN_SCORE = 80            # rapidfuzz token_sort_ratio, 0-100
CONFIDENCE_THRESHOLD = 0.75          # minimum combined confidence to accept a pairwise match
SPLIT_SETTLEMENT_MIN_CONFIDENCE = 0.70
SPLIT_SETTLEMENT_DATE_WINDOW_DAYS = 2
SPLIT_SETTLEMENT_NAME_MIN_SCORE = 90  # split-settlement rows keep the same merchant name verbatim


def normalize_id(txn_id: str) -> str:
    """Undo the case/dash/prefix reformats id_truncated and near_match_name
    apply: strip the TXN prefix, dashes, and case. Does NOT try to align
    truncation here - truncate_left drops the front 4 digits and
    truncate_right drops the back 4 digits of the same 12-digit number, so
    there is no single fixed-length slice that aligns both directions
    against the untruncated ledger id. See ids_match() for the actual
    comparison, which checks substring containment instead."""
    if not isinstance(txn_id, str):
        return ""
    return txn_id.upper().replace("TXN", "").replace("-", "")


def ids_match(*txn_ids: str) -> bool:
    """
    True iff every non-empty normalized id is consistent with a single
    underlying digit string - i.e. each one is a prefix, suffix, or exact
    match of the longest one among them. This is what actually lets
    truncate_left (drops the front) and truncate_right (drops the back)
    both align against an untruncated 12-digit ledger id, since a fixed-
    length slice cannot represent both truncation directions at once.
    Requires at least 7 digits of overlap so a short truncated id can't
    coincidentally satisfy substring containment against an unrelated id
    (1-in-10-million collision odds on random 7-digit strings, and this
    check never runs alone - callers additionally require amount+date
    agreement). 7 is the floor because truncate_right in the generator
    keeps txn_id[:10] including the 3-char "TXN" prefix, leaving exactly 7
    digits after the prefix is stripped - the shortest surviving id this
    dataset produces.
    """
    normed = [normalize_id(t) for t in txn_ids]
    if any(len(n) == 0 for n in normed):
        return False
    longest = max(normed, key=len)
    min_len = min(len(n) for n in normed)
    if min_len < 7:
        return False
    return all(n == longest or longest.startswith(n) or longest.endswith(n) for n in normed)


@dataclass
class MatchResult:
    ledger_id: str
    gateway_id: str          # for split_settlement this holds ";"-joined ids
    bank_id: str
    amount: float
    tier: str = "tier2_fuzzy"
    match_type: str = ""     # id_truncated | timezone_shift | rounding_drift | near_match_name | split_settlement
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class TierOutcome:
    matches: list = field(default_factory=list)
    unmatched_ledger: pd.DataFrame = None
    unmatched_gateway: pd.DataFrame = None
    unmatched_bank: pd.DataFrame = None


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _score_pairwise_candidate(ledger_row, gw_row, bk_row) -> tuple:
    """
    Score one (ledger, gateway, bank) candidate triple across the four
    non-split scenarios. Returns (confidence, match_type, rationale) or
    (0.0, None, "") if no signal combination clears its bar.

    Each sub-check is independently gated (id normalization must match
    within its own logic, date must be within window, amount must be within
    tolerance) - confidence is not a single fuzzy blend of unrelated
    signals, because that is exactly what would let a `duplicate` row
    (same amount, same/nearby date, different-by-nothing) slip through on
    coincidental similarity. At least two of {id, amount, date} must be
    exact-or-near for ANY match type below; near_match_name additionally
    requires the name signal.
    """
    l_id_norm = normalize_id(ledger_row["txn_id"])
    id_match = ids_match(ledger_row["txn_id"], gw_row["gateway_txn_id"], bk_row["reference_no"])

    l_amt = round(float(ledger_row["amount"]), 2)
    g_amt = round(float(gw_row["amount"]), 2)
    b_amt = round(float(bk_row["amount"]), 2)
    amount_exact = (l_amt == g_amt == b_amt)
    amount_diff = max(abs(l_amt - g_amt), abs(l_amt - b_amt))
    amount_close = amount_diff <= AMOUNT_TOLERANCE_RUPEES

    l_date = _parse_date(ledger_row["txn_date"])
    g_date = _parse_date(gw_row["settlement_date"])
    b_date = _parse_date(bk_row["value_date"])
    date_exact = (l_date == g_date == b_date)
    date_diff = max(abs((l_date - g_date).days), abs((l_date - b_date).days))
    date_close = date_diff <= DATE_WINDOW_DAYS

    # id_truncated: normalized ids agree, amount+date exact (id was the only field mangled)
    if id_match and amount_exact and date_exact:
        return (0.97, "id_truncated", f"normalized txn_id agrees ({l_id_norm}); amount+date exact")

    # timezone_shift: raw id+amount exact, date within window but not exact
    raw_id_exact = (ledger_row["txn_id"] == gw_row["gateway_txn_id"] == bk_row["reference_no"])
    if raw_id_exact and amount_exact and not date_exact and date_close:
        return (0.93, "timezone_shift", f"raw txn_id+amount exact; dates within {date_diff}d window")

    # rounding_drift: raw id exact, date exact, amount within sub-rupee tolerance but not exact
    if raw_id_exact and date_exact and not amount_exact and amount_close:
        return (0.90, "rounding_drift", f"raw txn_id+date exact; amount drift Rs{amount_diff:.2f} within tolerance")

    # near_match_name: normalized id agrees, amount+date exact/close, AND name similarity clears bar
    name_score = max(
        fuzz.token_sort_ratio(str(ledger_row["merchant_name"]), str(gw_row["merchant_name"])),
        fuzz.token_sort_ratio(str(ledger_row["merchant_name"]), str(bk_row["narration"])),
    )
    if id_match and (amount_exact or amount_close) and (date_exact or date_close) and name_score >= NAME_FUZZY_MIN_SCORE:
        combined = 0.5 + 0.5 * (name_score / 100.0)
        return (min(combined, 0.92), "near_match_name",
                f"normalized txn_id agrees; name similarity {name_score:.0f}/100; amount+date within tolerance")

    return (0.0, None, "")


def run_pairwise_fuzzy_match(
    unmatched_ledger: pd.DataFrame, unmatched_gateway: pd.DataFrame, unmatched_bank: pd.DataFrame
) -> TierOutcome:
    """
    Resolves id_truncated, timezone_shift, rounding_drift, near_match_name.
    split_settlement is intentionally excluded here - see
    run_split_settlement_match().

    Candidate generation is O(L*G*B) which is fine at this scale (dozens of
    rows post-Tier-1); at production scale this would be pre-bucketed by a
    coarse amount/date key before scoring.
    """
    matches = []
    matched_l, matched_g, matched_b = set(), set(), set()

    candidates = []
    for li, lrow in unmatched_ledger.iterrows():
        for gi, grow in unmatched_gateway.iterrows():
            # cheap prefilter: amount must be within a generous band before
            # we bother with id-normalization/date/name scoring at all
            if abs(round(float(lrow["amount"]), 2) - round(float(grow["amount"]), 2)) > max(AMOUNT_TOLERANCE_RUPEES, 5.0):
                continue
            for bi, brow in unmatched_bank.iterrows():
                if abs(round(float(lrow["amount"]), 2) - round(float(brow["amount"]), 2)) > max(AMOUNT_TOLERANCE_RUPEES, 5.0):
                    continue
                confidence, match_type, rationale = _score_pairwise_candidate(lrow, grow, brow)
                if confidence >= CONFIDENCE_THRESHOLD:
                    candidates.append((confidence, li, gi, bi, match_type, rationale, lrow, grow, brow))

    # Ambiguity guard, mirroring Tier 1's discipline: if a ledger row has
    # more than one DISTINCT candidate (li, gi, bi) that clears threshold -
    # e.g. two gateway rows that are raw-identical to each other, which is
    # exactly the `duplicate` scenario's shape - do not let the greedy sort
    # silently pick a "winner". Two indistinguishable candidates is genuine
    # ambiguity, not evidence for either one; resolving it here would let a
    # duplicate row slip through as a confident match. Same guard applied
    # from the gateway and bank sides, since ambiguity can originate there
    # too (two bank legs vs. one gateway/ledger row).
    from collections import defaultdict

    by_ledger = defaultdict(set)
    by_gateway = defaultdict(set)
    by_bank = defaultdict(set)
    for confidence, li, gi, bi, match_type, rationale, lrow, grow, brow in candidates:
        by_ledger[li].add((gi, bi))
        by_gateway[gi].add((li, bi))
        by_bank[bi].add((li, gi))

    ambiguous_l = {li for li, s in by_ledger.items() if len(s) > 1}
    ambiguous_g = {gi for gi, s in by_gateway.items() if len(s) > 1}
    ambiguous_b = {bi for bi, s in by_bank.items() if len(s) > 1}

    # Greedy assignment, highest confidence first, among the unambiguous
    # candidates only.
    candidates.sort(key=lambda c: -c[0])
    for confidence, li, gi, bi, match_type, rationale, lrow, grow, brow in candidates:
        if li in matched_l or gi in matched_g or bi in matched_b:
            continue
        if li in ambiguous_l or gi in ambiguous_g or bi in ambiguous_b:
            continue
        matches.append(
            MatchResult(
                ledger_id=lrow["ledger_id"],
                gateway_id=grow["gateway_id"],
                bank_id=brow["bank_id"],
                amount=round(float(lrow["amount"]), 2),
                match_type=match_type,
                confidence=round(confidence, 4),
                rationale=rationale,
            )
        )
        matched_l.add(li)
        matched_g.add(gi)
        matched_b.add(bi)

    remaining_ledger = unmatched_ledger.drop(index=list(matched_l)).reset_index(drop=True)
    remaining_gateway = unmatched_gateway.drop(index=list(matched_g)).reset_index(drop=True)
    remaining_bank = unmatched_bank.drop(index=list(matched_b)).reset_index(drop=True)

    return TierOutcome(matches, remaining_ledger, remaining_gateway, remaining_bank)


def run_split_settlement_match(
    unmatched_ledger: pd.DataFrame, unmatched_gateway: pd.DataFrame, unmatched_bank: pd.DataFrame
) -> TierOutcome:
    """
    Subset-sum grouping: for each unmatched ledger row, look for a SET of
    unmatched gateway rows (same merchant name, within a date window of the
    ledger date) whose amounts sum exactly (to the paisa) to the ledger
    amount, AND a single unmatched bank row that raw-matches the ledger's
    txn_id + amount + date (the bank side is never split in this dataset -
    only gateway settlement splits - so the bank leg still needs an anchor).

    This is deliberately a distinct routine from the pairwise fuzzy scorer:
    it is a one-to-many grouping problem, not a 1:1 similarity score, and
    forcing it through the same scorer would either miss it entirely or
    accept a partial (2-of-3) grouping as if it were complete - exactly the
    partial-match failure mode called out for this scenario. A match is only
    emitted when the FULL set of gateway legs is found and sums exactly;
    partial subset matches are left unresolved rather than reported as a
    match with missing legs.
    """
    matches = []
    matched_l, matched_g, matched_b = set(), set(), set()

    gw_pool = unmatched_gateway.copy()

    for li, lrow in unmatched_ledger.iterrows():
        l_amount = round(float(lrow["amount"]), 2)
        l_date = _parse_date(lrow["txn_date"])
        l_merchant = str(lrow["merchant_name"])

        # candidate gateway rows: same merchant name (near-exact, since
        # split settlement legs keep the merchant name verbatim), within a
        # slightly wider date window (settlement legs can post a few
        # minutes to a day apart), each individually smaller than the total
        candidate_idx = []
        for gi, grow in gw_pool.iterrows():
            if gi in matched_g:
                continue
            g_date = _parse_date(grow["settlement_date"])
            if abs((g_date - l_date).days) > SPLIT_SETTLEMENT_DATE_WINDOW_DAYS:
                continue
            if fuzz.token_sort_ratio(l_merchant, str(grow["merchant_name"])) < SPLIT_SETTLEMENT_NAME_MIN_SCORE:
                continue
            if round(float(grow["amount"]), 2) >= l_amount:
                continue  # a single leg can't be >= the whole ledger amount
            candidate_idx.append(gi)

        if len(candidate_idx) < 2:
            continue  # need at least 2 legs to call it a split, otherwise it's just an unmatched single row

        group = _find_exact_subset_sum(gw_pool, candidate_idx, l_amount)
        if group is None:
            continue  # no exact-summing subset found - leave unresolved rather than guess a partial grouping

        # anchor the bank leg: raw txn_id root (before "-S" suffix, if any)
        # + amount + date must exact-match one unmatched bank row
        bank_match_idx = None
        for bi, brow in unmatched_bank.iterrows():
            if bi in matched_b:
                continue
            if round(float(brow["amount"]), 2) != l_amount:
                continue
            if _parse_date(brow["value_date"]) != l_date:
                continue
            if not ids_match(brow["reference_no"], lrow["txn_id"]):
                continue
            bank_match_idx = bi
            break

        if bank_match_idx is None:
            continue  # gateway legs summed correctly but no bank anchor - don't half-match

        gw_ids = [gw_pool.loc[gi, "gateway_id"] for gi in group]
        matches.append(
            MatchResult(
                ledger_id=lrow["ledger_id"],
                gateway_id=";".join(gw_ids),
                bank_id=unmatched_bank.loc[bank_match_idx, "bank_id"],
                amount=l_amount,
                match_type="split_settlement",
                confidence=SPLIT_SETTLEMENT_MIN_CONFIDENCE + 0.2,
                rationale=(f"{len(group)} gateway legs ({', '.join(gw_ids)}) sum exactly to ledger amount "
                           f"Rs{l_amount:.2f}; same merchant name; bank anchor matched on id+amount+date"),
            )
        )
        matched_l.add(li)
        matched_g.update(group)
        matched_b.add(bank_match_idx)

    remaining_ledger = unmatched_ledger.drop(index=list(matched_l)).reset_index(drop=True)
    remaining_gateway = unmatched_gateway.drop(index=list(matched_g)).reset_index(drop=True)
    remaining_bank = unmatched_bank.drop(index=list(matched_b)).reset_index(drop=True)

    return TierOutcome(matches, remaining_ledger, remaining_gateway, remaining_bank)


def _find_exact_subset_sum(gw_pool: pd.DataFrame, candidate_idx: list, target: float, max_group_size: int = 4):
    """
    Find a subset of candidate_idx whose 'amount' values sum exactly
    (rounded to paisa) to target. Small N (candidate pools are a handful of
    rows post-Tier-1-and-prefilter) so brute-force subset enumeration up to
    max_group_size is cheap and exact - no need for DP/heuristics here.
    Returns the first exact-sum subset found, preferring smaller groups
    (checked in increasing size order) since real split settlements in this
    dataset are 2-3 legs.
    """
    from itertools import combinations

    target_paisa = round(target * 100)
    amounts_paisa = {gi: round(float(gw_pool.loc[gi, "amount"]) * 100) for gi in candidate_idx}

    for size in range(2, min(max_group_size, len(candidate_idx)) + 1):
        for combo in combinations(candidate_idx, size):
            if sum(amounts_paisa[gi] for gi in combo) == target_paisa:
                return list(combo)
    return None


def run_tier2(ledger: pd.DataFrame, gateway: pd.DataFrame, bank: pd.DataFrame):
    """Run both Tier 2 routines in sequence: split-settlement first (it has
    a narrower, more specific signature), then the pairwise fuzzy scorer on
    whatever remains."""
    split_outcome = run_split_settlement_match(ledger, gateway, bank)
    fuzzy_outcome = run_pairwise_fuzzy_match(
        split_outcome.unmatched_ledger, split_outcome.unmatched_gateway, split_outcome.unmatched_bank
    )

    all_matches = split_outcome.matches + fuzzy_outcome.matches
    combined = TierOutcome(
        matches=all_matches,
        unmatched_ledger=fuzzy_outcome.unmatched_ledger,
        unmatched_gateway=fuzzy_outcome.unmatched_gateway,
        unmatched_bank=fuzzy_outcome.unmatched_bank,
    )
    return combined, split_outcome, fuzzy_outcome


def verify_against_ground_truth(outcome: TierOutcome, ground_truth_path: Path = DATA_DIR / "ground_truth.csv"):
    """
    Same discipline as Tier 1's verifier: score matches against hidden
    ground truth, but split out:
      - overall Tier 2 precision/recall
      - split_settlement precision/recall reported SEPARATELY (it can
        partially succeed - e.g. 2-of-3 legs - which a lumped "Tier 2
        accuracy" number would hide)
      - explicit duplicate-leak and orphan-leak checks
      - scenario breakdown of what got matched, same shape as Tier 1's
    """
    gt = pd.read_csv(ground_truth_path)

    ledger_to_event, gateway_to_event, bank_to_event = {}, {}, {}
    event_scenario = {}
    for _, row in gt.iterrows():
        event_scenario[row["event_id"]] = row["scenario"]
        if pd.notna(row["ledger_ids"]) and row["ledger_ids"]:
            for lid in str(row["ledger_ids"]).split(";"):
                ledger_to_event[lid] = row["event_id"]
        if pd.notna(row["gateway_ids"]) and row["gateway_ids"]:
            for gid in str(row["gateway_ids"]).split(";"):
                gateway_to_event[gid] = row["event_id"]
        if pd.notna(row["bank_ids"]) and row["bank_ids"]:
            for bid in str(row["bank_ids"]).split(";"):
                bank_to_event[bid] = row["event_id"]

    def is_correct(m: MatchResult):
        l_evt = ledger_to_event.get(m.ledger_id)
        g_evts = {gateway_to_event.get(g) for g in m.gateway_id.split(";")}
        b_evt = bank_to_event.get(m.bank_id)
        return l_evt is not None and g_evts == {l_evt} and b_evt == l_evt, l_evt

    correct, incorrect = 0, 0
    incorrect_examples = []
    matched_scenarios = []
    duplicate_leaks, orphan_leaks = [], []

    split_correct, split_incorrect = 0, 0
    split_incorrect_examples = []

    for m in outcome.matches:
        ok, l_evt = is_correct(m)
        scenario = event_scenario.get(l_evt) if l_evt else None

        if m.match_type == "split_settlement":
            if ok:
                split_correct += 1
            else:
                split_incorrect += 1
                split_incorrect_examples.append(vars(m))

        if ok:
            correct += 1
            matched_scenarios.append(scenario)
            if scenario == "duplicate":
                duplicate_leaks.append(vars(m))
            if scenario == "orphan":
                orphan_leaks.append(vars(m))
        else:
            incorrect += 1
            incorrect_examples.append({**vars(m), "ledger_true_event": l_evt, "ledger_true_scenario": scenario})
            if scenario == "duplicate":
                duplicate_leaks.append(vars(m))
            if scenario == "orphan":
                orphan_leaks.append(vars(m))

    total = len(outcome.matches)
    precision = correct / total if total else float("nan")

    # Recall: of the events that Tier 2 is actually responsible for
    # (the 5 in-scope scenarios), how many did it correctly resolve.
    in_scope_scenarios = {"id_truncated", "timezone_shift", "rounding_drift", "near_match_name", "split_settlement"}
    n_in_scope_events = (gt["scenario"].isin(in_scope_scenarios)).sum()
    recall = correct / n_in_scope_events if n_in_scope_events else float("nan")

    n_split_events = (gt["scenario"] == "split_settlement").sum()
    split_precision = split_correct / (split_correct + split_incorrect) if (split_correct + split_incorrect) else float("nan")
    split_recall = split_correct / n_split_events if n_split_events else float("nan")

    scenario_counts = pd.Series(matched_scenarios).value_counts().to_dict() if matched_scenarios else {}

    return {
        "total_tier2_matches": total,
        "correct_matches": correct,
        "incorrect_matches": incorrect,
        "precision": precision,
        "recall": recall,
        "n_in_scope_events": int(n_in_scope_events),
        "incorrect_examples": incorrect_examples,
        "matched_scenario_breakdown": scenario_counts,
        "split_settlement": {
            "correct": split_correct,
            "incorrect": split_incorrect,
            "precision": split_precision,
            "recall": split_recall,
            "n_true_split_events": int(n_split_events),
            "incorrect_examples": split_incorrect_examples,
        },
        "duplicate_leaks": duplicate_leaks,
        "orphan_leaks": orphan_leaks,
    }


def main():
    from exact_matcher import load_sources, run_exact_match

    ledger, gateway, bank = load_sources()
    tier1_outcome = run_exact_match(ledger, gateway, bank)

    print(f"Tier 1 resolved {len(tier1_outcome.matches)} events; "
          f"{len(tier1_outcome.unmatched_ledger)} ledger rows remain for Tier 2.")
    print()

    combined, split_outcome, fuzzy_outcome = run_tier2(
        tier1_outcome.unmatched_ledger, tier1_outcome.unmatched_gateway, tier1_outcome.unmatched_bank
    )

    n_events_total = len(pd.read_csv(DATA_DIR / "ground_truth.csv"))
    print(f"Tier 2 (fuzzy/algorithmic): {len(combined.matches)} matches "
          f"({len(split_outcome.matches)} split_settlement + {len(fuzzy_outcome.matches)} pairwise fuzzy)")
    print(f"Unmatched after Tier 2: ledger={len(combined.unmatched_ledger)}, "
          f"gateway={len(combined.unmatched_gateway)}, bank={len(combined.unmatched_bank)}")

    report = verify_against_ground_truth(combined)
    print()
    print("--- Ground truth verification (Tier 2 overall) ---")
    print(f"Precision: {report['correct_matches']}/{report['total_tier2_matches']} = {report['precision']:.4%}")
    print(f"Recall (of {report['n_in_scope_events']} in-scope events): {report['recall']:.4%}")
    print(f"Matched scenario breakdown: {report['matched_scenario_breakdown']}")

    print()
    print("--- split_settlement (separate breakout) ---")
    ss = report["split_settlement"]
    print(f"Correct: {ss['correct']}/{ss['correct']+ss['incorrect']}  "
          f"Precision: {ss['precision']:.4%}  Recall: {ss['recall']:.4%} "
          f"(of {ss['n_true_split_events']} true split_settlement events)")

    print()
    print("--- Leak checks ---")
    print(f"duplicate-scenario events matched by Tier 2: {len(report['duplicate_leaks'])} (must be 0)")
    print(f"orphan-scenario events matched by Tier 2: {len(report['orphan_leaks'])} (must be 0)")

    if report["incorrect_matches"] > 0:
        print()
        print(f"!!! {report['incorrect_matches']} FALSE POSITIVE(S):")
        for ex in report["incorrect_examples"]:
            print(f"  {ex}")

    return combined, report


if __name__ == "__main__":
    main()
