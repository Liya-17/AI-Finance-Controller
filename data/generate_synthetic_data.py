"""
Synthetic data generator for the multi-source reconciliation agent.

Simulates three independent feeds that a banking-ops reconciliation team would
actually have to stitch together:

  1. core_ledger.csv        - internal core banking ledger entries
  2. gateway_settlement.csv - payment gateway settlement file
  3. bank_statement.csv     - bank statement / NEFT-RTGS feed

Each row across the three files originates from a single underlying
"transaction event". Instead of emitting clean 1:1:1 rows, we simulate the
failure modes that make reconciliation a real problem:

  - id_truncated     : transaction ID gets truncated/reformatted crossing systems
  - timezone_shift    : same transaction logged on a different calendar date
  - split_settlement  : one ledger entry settles as multiple gateway entries
  - rounding_drift    : sub-rupee paise-level amount mismatch (rounding, not fees)
  - duplicate         : a source double-writes the same event
  - near_match_name   : merchant/reference name typo'd or abbreviated, AND
                         the txn_id on that same side is reformatted (so this
                         can't slip through Tier 1 as a coincidental exact
                         id+amount match - the name corruption is the
                         intended signal, but the id reformat is what
                         actually routes it to Tier 2)
  - orphan            : event genuinely missing from one or more sources

Ground truth (which raw rows truly belong to the same event, and which
failure mode was applied) is written to ground_truth.csv and is NOT meant to
be consulted by the matchers - only by reports/metrics_report.md scoring.

SCENARIO -> RESOLUTION TIER CONTRACT
-------------------------------------
This mapping is the intended design contract for the matcher pipeline
(src/matchers/*). It exists here, next to the scenario definitions, so the
tiers are built deliberately against known ground truth rather than guessed:

  clean             -> Tier 1 (exact match): identical ID + amount + date,
                        no normalization needed.
  id_truncated      -> Tier 2 (ID normalization): deterministic but not
                        raw-string-exact - strip prefix / case-fold / de-dash
                        before exact compare.
  near_match_name   -> Tier 2 (fuzzy string similarity): scored on
                        merchant/reference name. Also carries a reformatted
                        txn_id on the same side (so it cannot pass Tier 1 on
                        id+amount alone) - Tier 2 should resolve it via name
                        similarity + id normalization + amount/date agreement
                        together, not name alone.
  timezone_shift    -> Tier 2 (date-window tolerance): deterministic logic,
                        just needs a +/-1 day window, not fuzzy text matching.
  rounding_drift    -> Tier 2 (amount-tolerance matching): sub-rupee band
                        (e.g. +/- Re 1), a clean tolerance-band case.
  split_settlement  -> Tier 2 (subset-sum grouping): NOT a 1:1 fuzzy match.
                        Requires finding N gateway rows whose amounts sum to
                        the ledger amount within a merchant + date window.
  duplicate         -> Tier 3 (LLM adjudication): genuinely ambiguous - two
                        rows with identical amount could be a true duplicate
                        or two legitimate separate payments that happen to
                        match. Requires judgment, not a rule.
  orphan            -> Unresolved / exceptions bucket. Has no true match in
                        one source by construction. No tier should ever
                        force-match this - doing so would inflate the match
                        rate on a case that is correctly unmatchable.
"""

import argparse
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# Realistic Indian business name pool: mix of Pvt Ltd / LLP / proprietorship
# style names across recognizable categories (retail, logistics, D2C,
# food & FMCG, textiles, electronics, services) - used instead of Faker's
# default (Western-sounding) company() generator, since this dataset is
# explicitly grounded in Indian banking rails (IMPS/NEFT/UPI).
INDIAN_MERCHANT_PREFIXES = [
    "Shree", "Sri", "Om", "Jai", "Maa", "Krishna", "Ganesh", "Laxmi", "Balaji",
    "Annapurna", "Shiv", "Royal", "National", "Modern", "Classic", "Prime",
    "Metro", "Sunrise", "Vishwa", "Bharat", "Nova", "Vertex", "Sundar",
]
INDIAN_MERCHANT_CORES = [
    "Traders", "Retail", "Mart", "Logistics", "Textiles", "Enterprises",
    "Electronics", "Foods", "Agro", "Apparels", "Distributors", "Exports",
    "Fashions", "Hardware", "Industries", "Motors", "Pharma", "Plastics",
    "Fresh Foods", "Wellness", "Fintech", "Commerce", "Packaging", "Grocers",
    "Handlooms", "Furnishings", "Bakers", "Snacks", "Beverages", "Rides",
    "Interiors", "Timber", "Steel", "Cables", "Solutions", "Automation",
]
INDIAN_MERCHANT_SUFFIXES = [
    "Pvt Ltd", "Private Limited", "LLP", "& Sons", "& Co", "Enterprises",
    "Industries", "Trading Co", "",  # trailing "" gives proprietorship-style bare names
]
INDIAN_PERSON_SURNAMES = [
    "Sharma", "Verma", "Gupta", "Reddy", "Nair", "Iyer", "Patel", "Shah",
    "Mehta", "Rao", "Kulkarni", "Joshi", "Chatterjee", "Bose", "Menon",
    "Pillai", "Agarwal", "Bansal", "Chawla", "Kapoor", "Malhotra", "Rana",
]


def make_indian_merchant_name(rng: random.Random) -> str:
    """Generate a realistic Indian business name (not a real company)."""
    style = rng.choice(["prefix_core_suffix", "person_and_sons", "person_core"])
    if style == "person_and_sons":
        surname = rng.choice(INDIAN_PERSON_SURNAMES)
        suffix = rng.choice(["& Sons", "& Co", "Brothers", "& Sons Pvt Ltd"])
        return f"{surname} {suffix}"
    if style == "person_core":
        surname = rng.choice(INDIAN_PERSON_SURNAMES)
        core = rng.choice(INDIAN_MERCHANT_CORES)
        suffix = rng.choice([s for s in INDIAN_MERCHANT_SUFFIXES if s != core])
        return f"{surname} {core}" + (f" {suffix}" if suffix else "")
    prefix = rng.choice(INDIAN_MERCHANT_PREFIXES)
    core = rng.choice(INDIAN_MERCHANT_CORES)
    suffix = rng.choice([s for s in INDIAN_MERCHANT_SUFFIXES if s != core])
    return f"{prefix} {core}" + (f" {suffix}" if suffix else "")

FAILURE_MODES = [
    "id_truncated",
    "timezone_shift",
    "split_settlement",
    "rounding_drift",
    "duplicate",
    "near_match_name",
    "orphan",
]

MERCHANT_ABBREVIATIONS = {
    "Private Limited": ["Pvt Ltd", "Pvt. Ltd.", "P Ltd"],
    "Pvt Ltd": ["Pvt. Ltd.", "P Ltd", "Private Limited"],
    "Enterprises": ["Enterp", "Entp", "Ent"],
    "Trading Co": ["Trdg Co", "Trading Company"],
    "Distributors": ["Distrib", "Dist"],
    "Electronics": ["Elec", "Electronic"],
    "Industries": ["Inds", "Indus"],
    "& Sons": ["and Sons", "&Sons"],
    "& Co": ["and Co", "&Co"],
    "Traders": ["Trdrs", "Trading"],
    "Logistics": ["Logistic", "Log"],
}

TXN_MODES = ["UPI", "NEFT", "RTGS", "IMPS", "CARD"]


def make_txn_id(rng: random.Random) -> str:
    return "TXN" + "".join(rng.choices(string.digits, k=12))


def truncate_or_reformat_id(txn_id: str, rng: random.Random) -> str:
    """Simulate an ID getting mangled crossing systems."""
    variant = rng.choice(["truncate_left", "truncate_right", "strip_prefix", "dashes", "lowercase"])
    if variant == "truncate_left":
        return txn_id[-10:]
    if variant == "truncate_right":
        return txn_id[:10]
    if variant == "strip_prefix":
        return txn_id.replace("TXN", "")
    if variant == "dashes":
        # TXN123456789012 -> TXN-1234-5678-9012
        digits = txn_id[3:]
        return f"TXN-{digits[0:4]}-{digits[4:8]}-{digits[8:12]}"
    if variant == "lowercase":
        return txn_id.lower()
    return txn_id


def abbreviate_merchant(name: str, rng: random.Random) -> str:
    """Introduce a near-match variant of a merchant/reference name."""
    result = name
    applied = False
    for full, variants in MERCHANT_ABBREVIATIONS.items():
        if full in result and rng.random() < 0.7:
            result = result.replace(full, rng.choice(variants))
            applied = True
    if not applied:
        # fall back to a light typo: drop or swap one character
        if len(result) > 4:
            idx = rng.randrange(1, len(result) - 1)
            chars = list(result)
            if rng.random() < 0.5:
                del chars[idx]
            else:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            result = "".join(chars)
    return result


def jitter_datetime(dt: datetime, rng: random.Random, mode: str) -> datetime:
    """
    mode='timezone_shift': push across a calendar-day boundary (e.g. UTC vs IST
    logging, or end-of-day batch cutoffs). GUARANTEED to change dt.date() -
    the offset is derived from how far dt actually sits from the nearest
    midnight, not a fixed hour pool, so this scenario can never silently
    land back on the same calendar date (which would make it
    indistinguishable from `clean` to any date-based check, including
    Tier 1's raw-date equality).
    mode='minor': a few minutes of natural processing delay (always applied,
    independent of injected failure modes, since real systems never log the
    exact same millisecond). Clamped so it never crosses a calendar-day
    boundary - unintended date-crossing is specifically what the
    `timezone_shift` scenario models; letting ordinary processing noise
    cross midnight too would make `clean` events spuriously fail Tier 1's
    date-equality check, contaminating the one scenario meant to be a clean
    baseline.
    """
    if mode == "timezone_shift":
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        seconds_in_day = 24 * 3600
        push_forward = rng.random() < 0.5
        if push_forward:
            # push forward past the next midnight
            min_seconds = seconds_in_day - seconds_since_midnight + 1
            offset = rng.randint(min_seconds, min_seconds + 6 * 3600)
            return dt + timedelta(seconds=offset)
        else:
            # push backward past the previous midnight
            min_seconds = seconds_since_midnight + 1
            offset = rng.randint(min_seconds, min_seconds + 6 * 3600)
            return dt - timedelta(seconds=offset)
    # minor natural processing delay between systems, clamped per-direction
    # so it can never cross into the next/previous calendar day
    seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
    seconds_until_midnight = 24 * 3600 - seconds_since_midnight - 1
    forward = rng.random() < 0.7  # delay is usually forward, occasionally an earlier log
    max_delay = min(240 * 60, seconds_until_midnight if forward else seconds_since_midnight)
    delay = rng.randint(0, max(max_delay, 0))
    return dt + timedelta(seconds=delay if forward else -delay)


def rounding_drift_amount(amount: float, rng: random.Random) -> float:
    """
    Simulate genuine sub-rupee rounding drift (e.g. two systems rounding a
    paise-level figure differently). Strictly < Re 1 so this stays a true
    "rounding" case rather than reading like a fee deduction or data error -
    Tier 2's amount-tolerance band is calibrated against this range.
    """
    drift_paise = rng.choice([1, 2, 5, 10, 15, 20, 25, 33, 49, 50, 67, 75, 90, 99])
    sign = rng.choice([-1, 1])
    new_amount = round(amount + sign * drift_paise / 100.0, 2)
    return max(new_amount, 0.01)


def gen_events(n_events: int, seed: int, injection_rate: float):
    """
    Build n_events underlying transaction events, each assigned exactly one
    scenario (clean or one of FAILURE_MODES), weighted by injection_rate.

    Returns list of event dicts describing the "true" transaction plus its
    assigned scenario, ready to be expanded into per-source rows.
    """
    rng = random.Random(seed)

    start_date = datetime(2026, 6, 1)
    events = []

    for i in range(n_events):
        event_id = f"EVT{i:06d}"
        amount = round(rng.uniform(150, 250000) / 100, 2) * 100 + round(rng.uniform(0, 99), 2)
        amount = round(amount, 2)
        merchant = make_indian_merchant_name(rng)
        txn_time = start_date + timedelta(
            days=rng.randint(0, 59), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        mode = rng.choice(TXN_MODES)
        base_txn_id = make_txn_id(rng)

        if rng.random() < injection_rate:
            scenario = rng.choice(FAILURE_MODES)
        else:
            scenario = "clean"

        events.append(
            {
                "event_id": event_id,
                "txn_id": base_txn_id,
                "amount": amount,
                "merchant": merchant,
                "txn_time": txn_time,
                "mode": mode,
                "scenario": scenario,
            }
        )
    return events


def expand_event_to_rows(event, rng: random.Random, ledger_rows, gateway_rows, bank_rows, ground_truth_rows):
    """
    Given one underlying event + its assigned scenario, emit the corresponding
    row(s) into the three source tables and record the true mapping.
    """
    scenario = event["scenario"]
    event_id = event["event_id"]
    txn_id = event["txn_id"]
    amount = event["amount"]
    merchant = event["merchant"]
    txn_time = event["txn_time"]
    mode = event["mode"]

    ledger_id = f"LDG{len(ledger_rows):06d}"
    gateway_id = f"GTW{len(gateway_rows):06d}"
    bank_id = f"BNK{len(bank_rows):06d}"

    # --- Ledger row (always present - it's the internal source of truth) ---
    ledger_time = jitter_datetime(txn_time, rng, "minor")
    ledger_rows.append(
        {
            "ledger_id": ledger_id,
            "txn_id": txn_id,
            "amount": amount,
            "merchant_name": merchant,
            "txn_date": ledger_time.strftime("%Y-%m-%d"),
            "txn_datetime": ledger_time.strftime("%Y-%m-%d %H:%M:%S"),
            "payment_mode": mode,
            "status": "POSTED",
        }
    )

    gateway_txn_id = txn_id
    bank_ref = txn_id
    gateway_amount = amount
    bank_amount = amount
    gateway_merchant = merchant
    bank_merchant = merchant
    gateway_time = jitter_datetime(txn_time, rng, "minor")
    bank_time = jitter_datetime(txn_time, rng, "minor")

    matched_gateway_ids = []
    matched_bank_ids = []

    if scenario == "orphan":
        # Missing from ONE of gateway/bank (chosen at random) - genuinely unresolved.
        drop = rng.choice(["gateway", "bank"])
        if drop != "gateway":
            gateway_rows.append(
                {
                    "gateway_id": gateway_id,
                    "gateway_txn_id": gateway_txn_id,
                    "amount": gateway_amount,
                    "merchant_name": gateway_merchant,
                    "settlement_date": gateway_time.strftime("%Y-%m-%d"),
                    "settlement_datetime": gateway_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "payment_mode": mode,
                    "gateway_status": "SETTLED",
                }
            )
            matched_gateway_ids.append(gateway_id)
        if drop != "bank":
            bank_rows.append(
                {
                    "bank_id": bank_id,
                    "reference_no": bank_ref,
                    "amount": bank_amount,
                    "narration": bank_merchant,
                    "value_date": bank_time.strftime("%Y-%m-%d"),
                    "value_datetime": bank_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "txn_mode": mode,
                }
            )
            matched_bank_ids.append(bank_id)

        ground_truth_rows.append(
            {
                "event_id": event_id,
                "scenario": scenario,
                "ledger_ids": ledger_id,
                "gateway_ids": ";".join(matched_gateway_ids),
                "bank_ids": ";".join(matched_bank_ids),
                "true_amount": amount,
                "notes": f"missing from {drop}",
            }
        )
        return

    if scenario == "id_truncated":
        gateway_txn_id = truncate_or_reformat_id(txn_id, rng)
        bank_ref = truncate_or_reformat_id(txn_id, rng)

    if scenario == "timezone_shift":
        # shift just one side's calendar date
        if rng.random() < 0.5:
            gateway_time = jitter_datetime(txn_time, rng, "timezone_shift")
        else:
            bank_time = jitter_datetime(txn_time, rng, "timezone_shift")

    if scenario == "rounding_drift":
        if rng.random() < 0.5:
            gateway_amount = rounding_drift_amount(amount, rng)
        else:
            bank_amount = rounding_drift_amount(amount, rng)

    if scenario == "near_match_name":
        # Corrupt the name AND reformat the txn_id on the same side. Without
        # the id reformat, txn_id+amount would still be raw-identical across
        # all three sources and this event would silently exact-match at
        # Tier 1 despite being a fuzzy-name scenario - the id reformat is
        # what actually routes it to Tier 2, same as id_truncated.
        if rng.random() < 0.5:
            gateway_merchant = abbreviate_merchant(merchant, rng)
            gateway_txn_id = truncate_or_reformat_id(txn_id, rng)
        else:
            bank_merchant = abbreviate_merchant(merchant, rng)
            bank_ref = truncate_or_reformat_id(txn_id, rng)

    # --- Gateway row(s) ---
    if scenario == "split_settlement":
        n_splits = rng.choice([2, 2, 3])
        splits = _split_amount(amount, n_splits, rng)
        for j, split_amt in enumerate(splits):
            gid = f"GTW{len(gateway_rows):06d}"
            split_time = gateway_time + timedelta(minutes=j * 5)
            gateway_rows.append(
                {
                    "gateway_id": gid,
                    "gateway_txn_id": f"{gateway_txn_id}-S{j+1}",
                    "amount": split_amt,
                    "merchant_name": gateway_merchant,
                    "settlement_date": split_time.strftime("%Y-%m-%d"),
                    "settlement_datetime": split_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "payment_mode": mode,
                    "gateway_status": "SETTLED",
                }
            )
            matched_gateway_ids.append(gid)
    else:
        gateway_rows.append(
            {
                "gateway_id": gateway_id,
                "gateway_txn_id": gateway_txn_id,
                "amount": gateway_amount,
                "merchant_name": gateway_merchant,
                "settlement_date": gateway_time.strftime("%Y-%m-%d"),
                "settlement_datetime": gateway_time.strftime("%Y-%m-%d %H:%M:%S"),
                "payment_mode": mode,
                "gateway_status": "SETTLED",
            }
        )
        matched_gateway_ids.append(gateway_id)

    # --- Bank row ---
    bank_rows.append(
        {
            "bank_id": bank_id,
            "reference_no": bank_ref,
            "amount": bank_amount,
            "narration": bank_merchant,
            "value_date": bank_time.strftime("%Y-%m-%d"),
            "value_datetime": bank_time.strftime("%Y-%m-%d %H:%M:%S"),
            "txn_mode": mode,
        }
    )
    matched_bank_ids.append(bank_id)

    # --- Duplicate: one source double-writes the same event ---
    if scenario == "duplicate":
        dup_source = rng.choice(["gateway", "bank"])
        if dup_source == "gateway":
            dup_id = f"GTW{len(gateway_rows):06d}"
            dup_time = gateway_time + timedelta(seconds=rng.randint(1, 120))
            gateway_rows.append(
                {
                    "gateway_id": dup_id,
                    "gateway_txn_id": gateway_txn_id,
                    "amount": gateway_amount,
                    "merchant_name": gateway_merchant,
                    "settlement_date": dup_time.strftime("%Y-%m-%d"),
                    "settlement_datetime": dup_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "payment_mode": mode,
                    "gateway_status": "SETTLED",
                }
            )
            matched_gateway_ids.append(dup_id)
        else:
            dup_id = f"BNK{len(bank_rows):06d}"
            dup_time = bank_time + timedelta(seconds=rng.randint(1, 120))
            bank_rows.append(
                {
                    "bank_id": dup_id,
                    "reference_no": bank_ref,
                    "amount": bank_amount,
                    "narration": bank_merchant,
                    "value_date": dup_time.strftime("%Y-%m-%d"),
                    "value_datetime": dup_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "txn_mode": mode,
                }
            )
            matched_bank_ids.append(dup_id)

    ground_truth_rows.append(
        {
            "event_id": event_id,
            "scenario": scenario,
            "ledger_ids": ledger_id,
            "gateway_ids": ";".join(matched_gateway_ids),
            "bank_ids": ";".join(matched_bank_ids),
            "true_amount": amount,
            "notes": "",
        }
    )


def _split_amount(total: float, n_splits: int, rng: random.Random):
    """Split an amount into n_splits positive parts summing exactly to total (rounded to paise)."""
    cuts = sorted(rng.uniform(0.1, 0.9) for _ in range(n_splits - 1))
    bounds = [0.0] + cuts + [1.0]
    parts = [round(total * (bounds[i + 1] - bounds[i]), 2) for i in range(n_splits)]
    # fix rounding remainder on the last part so parts sum exactly to total
    parts[-1] = round(total - sum(parts[:-1]), 2)
    return parts


def generate(n_records: int, injection_rate: float, seed: int, out_dir: Path):
    """
    n_records is interpreted as the approximate TOTAL row count across all
    three source files (matching the brief's "300-500 synthetic records
    total"). We back out the number of underlying events from that, since
    split settlements and duplicates add extra rows per event.
    """
    rng = random.Random(seed)
    # empirically ~2.85 rows/event on average given the scenario mix below;
    # solve for n_events so total rows lands near n_records.
    avg_rows_per_event = 2.85
    n_events = max(10, round(n_records / avg_rows_per_event))

    events = gen_events(n_events, seed, injection_rate)

    ledger_rows, gateway_rows, bank_rows, ground_truth_rows = [], [], [], []
    for event in events:
        expand_event_to_rows(event, rng, ledger_rows, gateway_rows, bank_rows, ground_truth_rows)

    # Shuffle row order within each source so matchers can't cheat off insertion order.
    rng.shuffle(ledger_rows)
    rng.shuffle(gateway_rows)
    rng.shuffle(bank_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_df = pd.DataFrame(ledger_rows)
    gateway_df = pd.DataFrame(gateway_rows)
    bank_df = pd.DataFrame(bank_rows)
    gt_df = pd.DataFrame(ground_truth_rows)

    ledger_df.to_csv(out_dir / "core_ledger.csv", index=False)
    gateway_df.to_csv(out_dir / "gateway_settlement.csv", index=False)
    bank_df.to_csv(out_dir / "bank_statement.csv", index=False)
    gt_df.to_csv(out_dir / "ground_truth.csv", index=False)

    total_rows = len(ledger_df) + len(gateway_df) + len(bank_df)
    scenario_counts = gt_df["scenario"].value_counts().to_dict()

    print(f"Generated {n_events} events -> {total_rows} total rows "
          f"(ledger={len(ledger_df)}, gateway={len(gateway_df)}, bank={len(bank_df)})")
    print(f"Injection rate: {injection_rate:.0%}")
    print("Scenario breakdown:")
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"  {scenario:<20s} {count:4d}  ({count / n_events:.1%})")
    print(f"\nWrote files to: {out_dir}")

    return ledger_df, gateway_df, bank_df, gt_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reconciliation test data.")
    parser.add_argument("--records", type=int, default=400, help="Approx total rows across all 3 sources (default 400)")
    parser.add_argument("--injection-rate", type=float, default=0.55, help="Fraction of events given a failure mode (default 0.55)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default=str(DATA_DIR), help="Output directory for CSVs")
    args = parser.parse_args()

    generate(args.records, args.injection_rate, args.seed, Path(args.out_dir))


if __name__ == "__main__":
    main()
