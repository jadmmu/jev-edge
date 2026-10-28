"""Recompute every numeric claim in JEV_on_Pi_Working_Manuscript.tex from the raw CSVs.

Read-only. Prints PASS/FAIL per claim plus the computed value.
"""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "q8_baseline_20261024T193824Z.csv"
RAND = ROOT / "results" / "q8_randomized_20261024T221759Z.csv"
PERT = ROOT / "results" / "q8_perturbation_20261025T040301Z.csv"
DSET = ROOT / "experiments" / "datasets" / "benchmark_20.jsonl"

fails, passes = [], []


def check(label, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    (passes if ok else fails).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} tex={want!r:<14} computed={float(got):.6f}")


def check_exact(label, got, want):
    ok = got == want
    (passes if ok else fails).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} tex={want!r:<14} computed={got!r}")


def load(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ms(rows):
    return np.array([float(r["wall_clock_ms"]) for r in rows])


def pct(v, q):
    return float(np.percentile(v, q, method="linear"))


print("=" * 100)
print("A. RECORD COUNTS")
print("=" * 100)
b_all, r_all, p_all = load(BASE), load(RAND), load(PERT)
b = [r for r in b_all if r["is_warmup"].strip().lower() == "false"]
r = [r for r in r_all if r["is_warmup"].strip().lower() == "false"]
p = [r for r in p_all if r["is_warmup"].strip().lower() == "false"]
check_exact("baseline measured n", len(b), 200)
check_exact("baseline warmups", len(b_all) - len(b), 20)
check_exact("randomized measured n", len(r), 200)
check_exact("randomized warmups", len(r_all) - len(r), 20)
check_exact("perturbation measured n", len(p), 240)
check_exact("perturbation warmups", len(p_all) - len(p), 24)
check_exact("total measured (conclusion: 640)", len(b) + len(r) + len(p), 640)

print()
print("=" * 100)
print("B. LATENCY THRESHOLD")
print("=" * 100)
THRESH = 1380.289
bv = np.sort(ms(b))
gaps = np.diff(bv)
i = int(np.argmax(gaps))
check("largest sorted-latency gap midpoint ~ threshold", (bv[i] + bv[i + 1]) / 2, THRESH, 0.05)
print(f"        largest gap is {gaps[i]:.3f} ms between {bv[i]:.3f} and {bv[i+1]:.3f}")

print()
print("=" * 100)
print("C. TABLE 1 — SEQUENTIAL COLUMN (n=200)")
print("=" * 100)
bms = ms(b)
check("sequential mean", bms.mean(), 1221.26, 0.005)
check("sequential median", pct(bms, 50), 365.28, 0.005)
check("sequential P90", pct(bms, 90), 2582.43, 0.005)
check("sequential P95", pct(bms, 95), 2612.05, 0.005)
check("sequential SD (ddof=0)", bms.std(ddof=0), 1066.73, 0.005)
print(f"        for reference SD(ddof=1) = {bms.std(ddof=1):.3f}")

print()
print("=" * 100)
print("D. TABLE 1 — RANDOMIZED COLUMN (n=200)")
print("=" * 100)
rms = ms(r)
check("randomized mean", rms.mean(), 1254.33, 0.005)
check("randomized median", pct(rms, 50), 386.57, 0.005)
check("randomized P90", pct(rms, 90), 2656.52, 0.005)
check("randomized P95", pct(rms, 95), 2687.84, 0.005)
check("randomized SD (ddof=0)", rms.std(ddof=0), 1090.39, 0.005)

print()
print("=" * 100)
print("E. FAST / SLOW PARTITIONS (baseline)")
print("=" * 100)
fast, slow = bms[bms < THRESH], bms[bms >= THRESH]
check_exact("fast partition count (tex: 120)", int(fast.size), 120)
check("fast partition mean (tex: 351.84)", fast.mean(), 351.84, 0.005)
check_exact("slow partition count (tex: 80)", int(slow.size), 80)
check("slow partition mean (tex: 2525.40)", slow.mean(), 2525.40, 0.005)

print()
print("=" * 100)
print("F. REGIME MEMBERSHIP AND PERSISTENCE")
print("=" * 100)


def by_q(rows):
    d = defaultdict(list)
    for x in rows:
        d[x["question_id"]].append(x)
    return d


bq, rq = by_q(b), by_q(r)
slow_q = sorted(q for q, rows in bq.items() if all(float(x["wall_clock_ms"]) >= THRESH for x in rows))
fast_q = sorted(q for q, rows in bq.items() if all(float(x["wall_clock_ms"]) < THRESH for x in rows))
check_exact("distinct question ids in baseline (tex: 20)", len(bq), 20)
check_exact("questions entirely in slow regime (tex: 8)", len(slow_q), 8)
check_exact("slow + fast accounts for all 20", len(slow_q) + len(fast_q), 20)
check_exact("all 10 baseline reps of the 8 slow questions are slow",
            all(len(bq[q]) == 10 for q in slow_q), True)
r_slow_q = sorted(q for q, rows in rq.items() if all(float(x["wall_clock_ms"]) >= THRESH for x in rows))
check_exact("same 8 questions slow after randomization", r_slow_q == slow_q, True)
mixed = [q for q, rows in rq.items() if not (all(float(x["wall_clock_ms"]) >= THRESH for x in rows)
                                             or all(float(x["wall_clock_ms"]) < THRESH for x in rows))]
check_exact("no question changed regime / no mixed question (tex: 0)", len(mixed), 0)

print()
print("=" * 100)
print("G. CHRONOLOGICAL ORDER AND REGRESSION")
print("=" * 100)


def trial_index(rows):
    return np.array([float(x["trial_index"]) for x in rows])


check("sequential chronological Pearson r (tex: -0.467)", np.corrcoef(trial_index(b), bms)[0, 1], -0.467, 0.0005)
check("randomized chronological Pearson r (tex: -0.0048)", np.corrcoef(trial_index(r), rms)[0, 1], -0.0048, 0.00005)

qs = sorted(rq)
X = []
for x in r:
    row = [1.0]
    row += [1.0 if x["question_id"] == q else 0.0 for q in qs[1:]]
    row += [float(x["block_position"]) if x["block_position"] not in ("", "None") else 0.0,
            float(x["block_number"]) if x["block_number"] not in ("", "None") else 0.0]
    X.append(row)
X = np.array(X)
y = rms
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
ss_res = ((y - pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
check("additive regression R^2 (tex: 0.9957)", 1 - ss_res / ss_tot, 0.9957, 0.00005)

print()
print("=" * 100)
print("H. REPRODUCIBILITY OF ANSWERS AND DISTRIBUTIONS")
print("=" * 100)


def ans_and_probs(row):
    resp = row["response_json"] or ""
    try:
        d = json.loads(resp)
    except Exception:
        return None, None
    team = ((d.get("answers") or {}).get("team") or {})
    return team.get("choice"), json.dumps(team.get("probabilities"), sort_keys=True)


same_choice = same_probs = 0
for q in sorted(bq):
    rows = bq[q]
    choices = {ans_and_probs(x)[0] for x in rows}
    probs = {ans_and_probs(x)[1] for x in rows}
    if len(choices) == 1:
        same_choice += 1
    if len(probs) == 1:
        same_probs += 1
check_exact("questions with identical selected answer across 10 reps", same_choice, 20)
check_exact("questions with identical probability distribution across 10 reps", same_probs, 20)

items = [json.loads(l) for l in open(DSET, encoding="utf-8") if l.strip()]
labeled = [i for i in items if i.get("ground_truth_status") == "available"]
check_exact("dataset questions (tex: twenty)", len(items), 20)
check_exact("labeled questions (tex: eighteen)", len(labeled), 18)
check_exact("unlabeled questions (tex: two)", len(items) - len(labeled), 2)

agree = 0
total_lab = 0
want = {i["id"]: i["ground_truth"]["team"] for i in labeled}
for q, rows in bq.items():
    if q in want:
        for x in rows:
            total_lab += 1
            if ans_and_probs(x)[0] == want[q]:
                agree += 1
check_exact("labeled baseline trials (tex: 180)", total_lab, 180)
check_exact("labeled baseline trials agreeing (tex: 180)", agree, 180)

print()
print("=" * 100)
print("I. PERTURBATION TRANSITIONS")
print("=" * 100)
cond = defaultdict(list)
for x in p:
    cond[(x["parent_question_id"], x["variant_type"])].append(x)
check_exact("distinct conditions in measured set (tex: 24)", len(cond), 24)
parents = sorted({k[0] for k in cond})
check_exact("parent questions (tex: four)", len(parents), 4)
variants = sorted({k[1] for k in cond})
check_exact("six conditions per parent", len(variants), 6)

crossed = []
for (par, vt) in sorted(cond):
    if vt == "control":
        continue
    ctrl = cond.get((par, "control"))
    if not ctrl:
        continue
    tm, cm = ms(cond[(par, vt)]).mean(), ms(ctrl).mean()
    if (tm >= THRESH) != (cm >= THRESH):
        crossed.append((par, vt, cm, tm))
check_exact("non-control variants crossing the threshold (tex: 2)", len(crossed), 2)
for par, vt, cm, tm in crossed:
    print(f"        {par} / {vt}: control {cm:.4f} -> variant {tm:.4f}")

ctrl1, var1 = cond[("q001_billing_duplicate", "control")], cond[("q001_billing_duplicate", "spelling")]
ctrl4, var4 = cond[("q004_invoice_copy", "control")], cond[("q004_invoice_copy", "paraphrase")]
check("Q001 control mean (tex: 2610.02)", ms(ctrl1).mean(), 2610.02, 0.005)
check("Q001 variant mean (tex: 390.45)", ms(var1).mean(), 390.45, 0.005)
d1 = ms(var1).mean() - ms(ctrl1).mean()
check("Q001 difference (tex: -2219.57)", d1, -2219.57, 0.005)
check("Q001 % change (tex: about -85%)", 100 * d1 / ms(ctrl1).mean(), -84.92, 0.6)
check("Q004 control mean (tex: 376.07)", ms(ctrl4).mean(), 376.07, 0.005)
check("Q004 variant mean (tex: 2531.42)", ms(var4).mean(), 2531.42, 0.005)
d4 = ms(var4).mean() - ms(ctrl4).mean()
check("Q004 difference (tex: +2155.35)", d4, 2155.35, 0.005)
check("Q004 % change (tex: about +573%)", 100 * d4 / ms(ctrl4).mean(), 573.0, 1.0)

check_exact("Q001 control input tokens (tex: 106)", sorted({int(x["input_tokens"]) for x in ctrl1}), [106])
check_exact("Q001 spelling input tokens (tex: 107)", sorted({int(x["input_tokens"]) for x in var1}), [107])
check_exact("Q004 control input tokens (tex: 101)", sorted({int(x["input_tokens"]) for x in ctrl4}), [101])
check_exact("Q004 paraphrase input tokens (tex: 99)", sorted({int(x["input_tokens"]) for x in var4}), [99])

def side(rows):
    return ["slow" if float(x["wall_clock_ms"]) >= THRESH else "fast" for x in rows]


check_exact("Q001 all 10 blocks land on the new side", len(set(side(var1))), 1)
check_exact("Q004 all 10 blocks land on the new side", len(set(side(var4))), 1)
check_exact("Q001 blocks (tex: ten)", len({x["block_number"] for x in var1}), 10)
check_exact("Q001 decision retained (billing)", sorted({ans_and_probs(x)[0] for x in var1}), ["billing"])
check_exact("Q004 decision retained (billing)", sorted({ans_and_probs(x)[0] for x in var4}), ["billing"])

rep = json.load(open(ROOT / "results/tables/q8_perturbation_analysis/q8_perturbation_analysis_report.json"))
pc = load(ROOT / "results/tables/q8_perturbation_analysis/q8_perturbation_paired_comparison.csv")
pcx = {(x["parent_question_id"], x["variant_type"]): x for x in pc}
check("Q001 max JS divergence bits (tex: 0.000202)", float(pcx[("q001_billing_duplicate", "spelling")]["max_js_divergence_bits"]), 0.000202, 1e-6)
check("Q004 max JS divergence bits (tex: 0.000326)", float(pcx[("q004_invoice_copy", "paraphrase")]["max_js_divergence_bits"]), 0.000326, 1e-6)
check("Q001 max abs probability shift (tex: 0.00777)", float(pcx[("q001_billing_duplicate", "spelling")]["max_probability_shift"]), 0.00777, 1e-5)
check("Q004 max abs probability shift (tex: 0.00176)", float(pcx[("q004_invoice_copy", "paraphrase")]["max_probability_shift"]), 0.00176, 1e-5)
lo1, hi1 = [float(v) for v in pcx[("q001_billing_duplicate", "spelling")]["paired_difference_approx_95ci_ms"].strip("[]").split(",")]
lo4, hi4 = [float(v) for v in pcx[("q004_invoice_copy", "paraphrase")]["paired_difference_approx_95ci_ms"].strip("[]").split(",")]
check("Q001 paired CI low (tex: -2288.53)", lo1, -2288.53, 0.005)
check("Q001 paired CI high (tex: -2150.61)", hi1, -2150.61, 0.005)
check("Q004 paired CI low (tex: 2083.75)", lo4, 2083.75, 0.005)
check("Q004 paired CI high (tex: 2226.95)", hi4, 2226.95, 0.005)
check_exact("perturbation failures (tex: none)", len([x for x in p if x["error"].strip()]), 0)

print()
print("=" * 100)
print("J. FIGURES REFERENCED BY THE MANUSCRIPT")
print("=" * 100)
tex = (ROOT / "JEV_on_Pi_Working_Manuscript.tex").read_text(encoding="utf-8")
import re
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
for f in figs:
    hit = list(ROOT.rglob(Path(f).name))
    status = "FOUND" if hit else "MISSING FROM REPO"
    if hit:
        status += " at " + str(hit[0].relative_to(ROOT))
    if not hit:
        fails.append("figure " + f)
    print(f"  {status:<58} {f}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"PASS {len(passes)}   FAIL {len(fails)}")
if fails:
    print("\nFAILED CLAIMS:")
    for f in fails:
        print("  -", f)
sys.exit(1 if fails else 0)
