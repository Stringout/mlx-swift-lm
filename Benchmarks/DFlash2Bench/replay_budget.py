# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Replay the DFlash2 loss budget on recorded per-round acceptance.

Reads DFlash2Bench JSON (speculative arms with `roundAccepts`), derives each
round's tokens at a narrower width w from a wider record as
tau_w = min(accepted, w - 1) + 1, and replays the gate
S = max(0, S + R - tau), fall back once S >= c (decided on rounds already
synced, so the round in flight still commits) against round costs R.

A turn's cost is in plain steps: R per round, one per token after the fallback.
For each (R, c) it prints the share of turns tripped, the gated over ungated
gain on turns with tau/R > 1.15, and the worst loss in plain steps on turns
with tau/R < 1. Acceptance at every R: gain ratio >= 0.97, and worst loss
<= c + 2(R - 1), the budget plus the round that crosses it and the round in
flight. The looser c + R is printed too; the crossing round can exceed it
whenever R > 2.

    uv run replay_budget.py record.json [more.json] [--width 5] [--recorded 16]
        [--budgets 6,8,12,16,20,24,32,40,48]
"""

import argparse
import json
import math


def turns(paths, width, recorded=None):
    """Per-round tokens at `width` for every speculative arm recorded at `recorded`, or any."""
    traces = []
    for path in paths:
        with open(path) as handle:
            report = json.load(handle)
        for turn in report["turns"]:
            for arm in turn["arms"]:
                accepts = arm.get("roundAccepts")
                if arm["kind"] != "speculative" or not accepts:
                    continue
                if recorded is not None and arm["block"] != recorded:
                    continue
                if width > arm["block"]:
                    raise SystemExit(f"width {width} is wider than the record's {arm['block']}")
                traces.append(
                    {
                        "label": turn["label"],
                        "block": arm["block"],
                        "taus": [min(a, width - 1) + 1 for a in accepts],
                        "accepts": accepts,
                    }
                )
    return traces


def replay(taus, cost, budget):
    """Plain-step cost of the gated turn, and whether it tripped."""
    loss = 0.0
    spent = 0.0
    emitted = 0
    for tau in taus:
        stops = loss >= budget
        spent += cost
        emitted += tau
        loss = max(0.0, loss + cost - tau)
        if stops:
            return spent + (sum(taus) - emitted), True
    return spent, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--recorded", type=int, help="only arms recorded at this block")
    parser.add_argument("--budgets", default="6,8,12,16,20,24,32,40,48")
    args = parser.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    costs = [1 + 0.25 * i for i in range(21)]

    record = turns(args.records, args.width, args.recorded)
    print(f"{len(record)} speculative turns, {sum(len(t['taus']) for t in record)} rounds")

    print("\ntau at width w, pooled over rounds: E[min(A, w - 1)] + 1")
    rounds = [a for t in record for a in t["accepts"]]
    recorded = sorted({t["block"] for t in record})
    print(f"recorded at block {recorded}")
    for w in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]:
        if w > min(recorded):
            continue
        tau = sum(min(a, w - 1) + 1 for a in rounds) / len(rounds)
        print(f"  w {w:2d}: tau {tau:.3f}")

    print(f"\ngate replay at width {args.width}")
    print(
        "R     c     tripped  gain ratio (tau/R>1.15: pooled, worst)  "
        "worst loss (tau/R<1)  c+R    checked bound c+2(R-1)  ok"
    )
    verdict = {}
    lowest = {}
    for budget in budgets:
        verdict[budget] = True
        for cost in costs:
            tripped = 0
            gated_cost = ungated_cost = high_tokens = 0.0
            worst_ratio = math.inf
            worst_loss = 0.0
            for trace in record:
                taus = trace["taus"]
                tokens = sum(taus)
                spent, trip = replay(taus, cost, budget)
                tripped += trip
                ungated = cost * len(taus)
                mean_tau = tokens / len(taus)
                if mean_tau / cost > 1.15:
                    gated_cost += spent
                    ungated_cost += ungated
                    high_tokens += tokens
                    worst_ratio = min(worst_ratio, ungated / spent)
                if mean_tau / cost < 1.0:
                    worst_loss = max(worst_loss, spent - tokens)
            pooled = ungated_cost / gated_cost if gated_cost else math.nan
            bound = budget + 2 * (cost - 1)
            ok = (math.isnan(pooled) or pooled >= 0.97) and worst_loss <= bound
            verdict[budget] &= ok
            if not math.isnan(pooled):
                lowest[budget] = min(lowest.get(budget, math.inf), pooled)
            print(
                f"{cost:<5.2f} {budget:<5.0f} {tripped / len(record):<8.2f} "
                f"{pooled:<7.4f} {worst_ratio if worst_ratio < math.inf else math.nan:<31.4f} "
                f"{worst_loss:<21.2f} {budget + cost:<6.2f} {bound:<23.2f} {'yes' if ok else 'NO'}"
            )
    print()
    for budget in budgets:
        print(f"c {budget:<4.0f} lowest pooled gain ratio {lowest.get(budget, math.nan):.4f}")
    passing = [b for b in budgets if verdict[b]]
    print(f"budgets meeting gain >= 0.97 and loss <= c + 2(R - 1) at every R: {passing}")
    print(f"smallest: {min(passing) if passing else 'none'}")


if __name__ == "__main__":
    main()
