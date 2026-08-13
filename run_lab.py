"""
run_lab.py — chunk size experiments, decomposition, re-ranking, and cost.

    python3 run_lab.py                    # the full grid, scored offline
    python3 run_lab.py --quick            # hybrid only, faster
    python3 run_lab.py --show Q06         # one question, in detail
    python3 run_lab.py --invent           # what happens when the model guesses

    python3 run_lab.py --judge azure      # a REAL model answers and scores
    python3 run_lab.py --judge azure --variance 3   # how much of it is noise

Two judges, and you need both:

    mock   the deterministic rules in ragas_lab/metrics.py. Free, offline,
           reproducible to the digit. Use it to explore the whole grid.
    azure  a real Azure OpenAI deployment generates the answers and scores
           them the way the RAGAS library does. Costs money, takes minutes,
           and will not give you the same number twice. Use it on the
           shortlist mock produced — and read ragas_lab/judge.py before you
           quote a single number from it.

Configure the azure judge in .env (copy .env.example). RAGAS_JUDGE sets the
default; --judge overrides it.

Writes to out/:
    ragas_report.md    the evaluation report for your ADR
    results.csv        every configuration, every metric
    per_question.csv   where each configuration wins and loses
    judge_cache.json   azure mode only — so a re-run is free
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ragas_lab"))

from corpus import all_docs, EVAL_SET                      # noqa: E402
from retrieval import build_chunks, Index, retrieve, CHUNKERS  # noqa: E402
from metrics import evaluate_case, mean                    # noqa: E402
from judge import build_judge, load_env, JudgeError        # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# Illustrative. Look up real numbers for your region before quoting anything.
PRICE_PER_1K_INPUT_TOKENS = 0.00015

# Six API calls per question in azure mode: generate, faithfulness,
# answer relevancy, context precision, context recall, correctness. Refusals
# and negative-control questions use fewer.
CALLS_PER_QUESTION = 6

# The shortlist azure mode runs unless you pass --full. Chosen from what mock
# mode makes obvious: structure chunking and hybrid retrieval are the live
# candidates, and rerank is the one knob worth paying a judge to settle.
AZURE_DEFAULT_GRID = [
    ("structure",  "hybrid", False, False),
    ("structure",  "hybrid", True,  False),
    ("fixed_512",  "hybrid", False, False),
    ("fixed_256",  "hybrid", False, False),
]


def run_config(docs, chunker: str, mode: str, top_k: int,
               rerank: bool, decomp: bool, invent: bool,
               judge=None, workers: int = 6) -> dict:
    chunks = build_chunks(docs, chunker)
    index = Index(chunks)

    retrieved = [retrieve(index, case["question"], mode, top_k,
                          use_rerank=rerank, use_decomp=decomp)
                 for case in EVAL_SET]

    if judge is None:
        scored = [evaluate_case(got, case, invents_when_missing=invent)
                  for got, case in zip(retrieved, EVAL_SET)]
    else:
        # Six sequential HTTP round-trips per question, twelve questions. Judge
        # the questions concurrently or take a coffee break for every config.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored = list(pool.map(lambda p: judge.evaluate_case(*p),
                                   zip(retrieved, EVAL_SET)))
        judge.save_cache()

    rows, tok_total = [], 0
    for case, got, (sc, answer) in zip(EVAL_SET, retrieved, scored):
        tok = sum(c.n_tokens for c in got)
        tok_total += tok
        rows.append({
            "qid": case["id"], "type": case["type"],
            **sc.to_dict(), "tokens": tok,
            "answer": answer,
            "retrieved": [c.id for c in got],
        })

    n = len(EVAL_SET)
    return {
        "chunker": chunker, "retrieval": mode, "top_k": top_k,
        "rerank": rerank, "decompose": decomp,
        "n_chunks": len(chunks),
        "avg_chunk_tokens": round(mean([c.n_tokens for c in chunks]), 1),
        "context_precision": round(mean([r["context_precision"] for r in rows]), 3),
        "context_recall": round(mean([r["context_recall"] for r in rows]), 3),
        "faithfulness": round(mean([r["faithfulness"] for r in rows]), 3),
        "answer_relevancy": round(mean([r["answer_relevancy"] for r in rows]), 3),
        "correct": sum(1 for r in rows if r["correct"]),
        "of": n,
        "tokens_per_query": round(tok_total / n, 1),
        "cost_per_1k_queries": round(tok_total / n / 1000 * PRICE_PER_1K_INPUT_TOKENS * 1000, 4),
        "rows": rows,
    }


def by_type(res: dict) -> dict[str, float]:
    agg: dict[str, list[int]] = {}
    for r in res["rows"]:
        agg.setdefault(r["type"], []).append(1 if r["correct"] else 0)
    return {k: round(sum(v) / len(v), 2) for k, v in agg.items()}


def show_one(docs, qid: str, top_k: int = 5, judge=None) -> int:
    case = next((c for c in EVAL_SET if c["id"] == qid), None)
    if not case:
        print(f"No such question: {qid}. Try {[c['id'] for c in EVAL_SET]}")
        return 1
    print(f"\n{case['id']}  [{case['type']}]  {case['question']}")
    if case.get("trap"):
        print(f"  TRAP: {case['trap']}")
    print(f"  Ground truth: {case['answer']}\n")
    for chunker in CHUNKERS:
        chunks = build_chunks(docs, chunker)
        idx = Index(chunks)
        for mode in ("keyword", "vector", "hybrid"):
            got = retrieve(idx, case["question"], mode, top_k)
            sc, ans = (judge.evaluate_case(got, case) if judge
                       else evaluate_case(got, case))
            print(f"  {chunker:<11} {mode:<8} "
                  f"prec {sc.context_precision:.2f}  rec {sc.context_recall:.2f}  "
                  f"faith {sc.faithfulness:.2f}  rel {sc.answer_relevancy:.2f}  "
                  f"{'OK ' if sc.correct else 'x  '}")
            print(f"              -> {ans[:96]}")
    if judge:
        judge.save_cache()
    return 0


def run_variance(docs, args, judge, repeats: int) -> int:
    """Judge the SAME configuration N times and report the spread.

    This is the single most important number in the lab, and it is not on the
    scorecard. If faithfulness moves 0.06 between two identical runs, then a
    0.04 difference between two configurations is not a result — it is noise
    wearing a decimal point.

    Cache is off here by necessity: a cached judge is perfectly reproducible
    and would report a spread of zero, which is a property of the cache and
    not of the judge.
    """
    judge.use_cache = False
    chunker, mode, rr, dc = AZURE_DEFAULT_GRID[0]
    print(f"\nVariance probe: {chunker} + {mode}, {repeats} identical runs, "
          f"temperature {judge.temperature}, cache OFF")
    print("-" * 78)
    print(f"{'run':<5} {'prec':>6} {'rec':>6} {'faith':>6} {'relev':>6} {'correct':>9}")
    runs = []
    for i in range(repeats):
        r = run_config(docs, chunker, mode, args.top_k, rr, dc, False, judge=judge)
        runs.append(r)
        print(f"{i + 1:<5} {r['context_precision']:>6.3f} {r['context_recall']:>6.3f} "
              f"{r['faithfulness']:>6.3f} {r['answer_relevancy']:>6.3f} "
              f"{r['correct']:>6}/{r['of']:<2}")

    print("-" * 78)
    lines = []
    for m in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
        vals = [r[m] for r in runs]
        spread = max(vals) - min(vals)
        lines.append((m, mean(vals), spread))
        print(f"{m:<20} mean {mean(vals):.3f}   spread {spread:.3f}   "
              f"{'<- differences smaller than this are noise' if spread > 0.01 else ''}")

    (OUT / "judge_variance.md").write_text("\n".join(
        [f"# Judge variance — {repeats} identical runs", "",
         f"Configuration: `{chunker} + {mode}`  |  deployment: "
         f"`{judge.deployment}`  |  temperature: {judge.temperature}  |  cache: off", "",
         "| Metric | Mean | Spread (max-min) |", "|---|---|---|"]
        + [f"| {m} | {mu:.3f} | {sp:.3f} |" for m, mu, sp in lines]
        + ["", "Any difference between two configurations that is smaller than the "
           "spread above is not evidence. Report it as a tie.", ""]))
    print(f"\nWritten: {OUT / 'judge_variance.md'}")
    print(f"Judge:   {judge.stats.summary()}")
    return 0


def write_report(results: list[dict], args, judge=None) -> None:
    cols = ["chunker", "retrieval", "top_k", "rerank", "decompose", "n_chunks",
            "avg_chunk_tokens", "context_precision", "context_recall",
            "faithfulness", "answer_relevancy", "correct", "of",
            "tokens_per_query", "cost_per_1k_queries"]
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in cols})

    with open(OUT / "per_question.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunker", "retrieval", "rerank", "decompose", "qid", "type",
                    "context_precision", "context_recall", "faithfulness",
                    "answer_relevancy", "correct", "tokens", "retrieved", "answer"])
        for r in results:
            for q in r["rows"]:
                w.writerow([r["chunker"], r["retrieval"], r["rerank"], r["decompose"],
                            q["qid"], q["type"], q["context_precision"],
                            q["context_recall"], q["faithfulness"],
                            q["answer_relevancy"], q["correct"], q["tokens"],
                            " ".join(q["retrieved"]),
                            # The answer column is the point of an azure run.
                            # A scorecard tells you a configuration failed;
                            # only the text tells you which claim it confused
                            # yours with.
                            q["answer"].replace("\n", " ")])

    best = max(results, key=lambda r: (r["correct"], r["context_recall"], -r["tokens_per_query"]))
    md = ["# RAGAS evaluation report — DecisionStream AI retrieval", "",
          f"Configurations evaluated: **{len(results)}**   |   "
          f"Questions: **{len(EVAL_SET)}**   |   top_k: **{args.top_k}**", ""]

    if judge is None:
        md += ["**Judge: `mock`** — metrics computed deterministically in "
               "`ragas_lab/metrics.py`. These numbers are reproducible to the digit: "
               "re-run this and nothing moves. That is a property of the harness, "
               "not of your pipeline.", "",
               "Re-run the shortlist with `--judge azure` before anything here reaches "
               "an ADR. A real judge scores generated answers, not simulated ones, and "
               "it disagrees.", ""]
    else:
        s = judge.stats
        md += [f"**Judge: `azure`** — deployment `{judge.deployment}`, "
               f"api-version `{judge.api_version}`, temperature `{judge.temperature}`.",
               "",
               "Answers were **generated by the model** from the retrieved passages, "
               "then scored by the model using the RAGAS decomposition "
               "(see `ragas_lab/judge.py`). Every number below is therefore a model "
               "output with variance, cost, and a dependency on a model version you "
               "do not control.", "",
               f"Judge cost for this run: **{s.summary()}**.", "",
               "> Two things to carry into the ADR. First, these scores are not "
               "reproducible — run `--judge azure --variance 3` to see the spread, and "
               "treat any gap smaller than that spread as a tie. Second, an upgrade to "
               "the judge deployment can move these numbers while your pipeline is "
               "untouched. Pin the judge version and make its upgrade a review trigger.", ""]
        if (OUT / "judge_variance.md").exists():
            md += ["Measured noise floor for this judge: see "
                   "[`judge_variance.md`](judge_variance.md). Compare every gap in the "
                   "table below against it before you call anything a difference.", ""]

    md += ["## All configurations", "",
          "| Chunker | Retrieval | Rerank | Decomp | Prec | Recall | Faith | Relev | Correct | Tokens/q | $/1k q |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (-x["correct"], -x["context_recall"])):
        md.append(
            f"| {r['chunker']} | {r['retrieval']} | {'Y' if r['rerank'] else '-'} | "
            f"{'Y' if r['decompose'] else '-'} | {r['context_precision']:.2f} | "
            f"{r['context_recall']:.2f} | {r['faithfulness']:.2f} | "
            f"{r['answer_relevancy']:.2f} | {r['correct']}/{r['of']} | "
            f"{r['tokens_per_query']:.0f} | ${r['cost_per_1k_queries']:.3f} |")

    md += ["", "## Where each configuration wins and loses", "",
           "Averages hide the interesting part. This table is correctness by "
           "question type.", "",
           "| Chunker | Retrieval | Rerank | Decomp | " +
           " | ".join(sorted({r["type"] for r in EVAL_SET})) + " |",
           "|---|---|---|---|" + "---|" * len(set(r["type"] for r in EVAL_SET))]
    types = sorted({r["type"] for r in EVAL_SET})
    for r in sorted(results, key=lambda x: -x["correct"]):
        t = by_type(r)
        md.append(f"| {r['chunker']} | {r['retrieval']} | {'Y' if r['rerank'] else '-'} | "
                  f"{'Y' if r['decompose'] else '-'} | " +
                  " | ".join(f"{t.get(x, 0):.2f}" for x in types) + " |")

    # Surface the ceiling effect explicitly — otherwise it looks like a bug.
    sizes = {r["chunker"]: r["n_chunks"] for r in results}
    identical = [a for a in ("fixed_512", "fixed_1024")
                 if sizes.get(a) == sizes.get("fixed_1024") and a != "fixed_1024"]
    if identical and sizes.get("fixed_512") == sizes.get("fixed_1024"):
        md += ["", "## Why 512 and 1024 give identical results here", "",
               f"Both produce **{sizes['fixed_1024']} chunks**. Most documents in this "
               "corpus are shorter than 512 tokens, so a 512-token and a 1024-token "
               "chunker never actually split them differently.",
               "",
               "This is not a bug and it is worth understanding: **chunk size stops "
               "being a variable once it exceeds your typical document length.** "
               "Tuning it further is effort spent on a parameter that has no effect "
               "on your data. Check your document length distribution before you run "
               "a chunk size experiment — if most documents fit in one chunk, the "
               "experiment has already answered itself.", ""]

    md += ["", "## Best configuration on this evaluation set", "",
           f"**{best['chunker']} + {best['retrieval']}"
           f"{' + rerank' if best['rerank'] else ''}"
           f"{' + decomposition' if best['decompose'] else ''}** — "
           f"{best['correct']}/{best['of']} correct, "
           f"recall {best['context_recall']:.2f}, precision {best['context_precision']:.2f}, "
           f"{best['tokens_per_query']:.0f} tokens per query.", "",
           "## Questions to answer before this goes in the ADR", "",
           "1. Which question TYPE does your chosen configuration handle worst, "
           "and what happens in production when that type arrives?",
           "2. Recall and precision move in opposite directions as chunk size "
           "grows. Which one does your client's risk appetite favour, and who agreed that?",
           "3. What does the cost column look like at your real query volume, "
           "not at twelve questions?",
           "4. The version-sensitive questions (Q09, Q10) exist because the policy "
           "corpus contains two versions of the same clause. Did your configuration "
           "retrieve the right one — and would you have noticed if it had not?",
           "5. What is your review trigger for re-running this evaluation?", ""]

    (OUT / "ragas_report.md").write_text("\n".join(md))


def main() -> int:
    load_env(HERE / ".env")

    ap = argparse.ArgumentParser(description="RAG evaluation and optimisation lab")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="hybrid retrieval only")
    ap.add_argument("--show", metavar="QID", help="one question, all configs, in detail")
    ap.add_argument("--invent", action="store_true",
                    help="make the simulated model guess instead of refusing "
                         "(mock judge only — a real model needs no help)")
    ap.add_argument("--judge", choices=["mock", "azure"],
                    default=os.environ.get("RAGAS_JUDGE", "mock"),
                    help="mock = deterministic rules (default); "
                         "azure = a real deployment generates and scores")
    ap.add_argument("--full", action="store_true",
                    help="azure: run the whole grid instead of the shortlist. "
                         "Expensive. Read the call estimate before you agree to it.")
    ap.add_argument("--no-cache", action="store_true",
                    help="azure: do not reuse cached judgements")
    ap.add_argument("--temperature", type=float, default=None,
                    help="azure: override the judge temperature")
    ap.add_argument("--variance", type=int, metavar="N",
                    help="azure: judge one configuration N times and report the "
                         "spread, so you know which differences are real")
    ap.add_argument("--workers", type=int, default=6,
                    help="azure: questions judged concurrently (default 6)")
    args = ap.parse_args()

    docs = all_docs()
    print("DecisionStream AI — RAG evaluation lab")
    print("=" * 78)
    print(f"  documents: {len(docs)}   questions: {len(EVAL_SET)}   top_k: {args.top_k}")

    try:
        judge = build_judge(args.judge, HERE,
                            use_cache=not args.no_cache,
                            temperature=args.temperature)
    except JudgeError as e:
        print(f"\nJUDGE ERROR\n{e}")
        return 2

    if judge is None:
        print("  judge: mock (deterministic, offline, free)")
        if args.invent:
            print("  MODE: model INVENTS when context is incomplete (watch faithfulness)")
    else:
        print(f"  judge: azure — {judge.deployment} @ {judge.endpoint}")
        print(f"         temperature {judge.temperature}, "
              f"cache {'off' if args.no_cache else 'on'}, "
              f"budget {judge.max_calls} calls")
        if args.invent:
            print("  NOTE: --invent is a mock-judge switch and is ignored here. "
                  "A real model invents on its own or refuses on its own; that is "
                  "the behaviour you are measuring.")

    if args.show:
        return show_one(docs, args.show.upper(), args.top_k, judge)

    if judge is not None and args.variance:
        return run_variance(docs, args, judge, args.variance)

    # ---- pick the grid -------------------------------------------------
    if judge is not None and not args.full:
        grid = list(AZURE_DEFAULT_GRID)
        print(f"\n  grid: azure shortlist, {len(grid)} configurations "
              f"(--full runs all 48, and costs ~12x this)")
    else:
        chunkers = list(CHUNKERS)
        modes = ["hybrid"] if args.quick else ["keyword", "vector", "hybrid"]
        variants = [(False, False), (True, False), (False, True), (True, True)]
        grid = [(c, m, rr, dc) for c, m, (rr, dc)
                in itertools.product(chunkers, modes, variants)]

    if judge is not None:
        est = len(grid) * len(EVAL_SET) * CALLS_PER_QUESTION
        print(f"  estimate: up to {est} API calls "
              f"({len(grid)} configs x {len(EVAL_SET)} questions x "
              f"{CALLS_PER_QUESTION} calls), before cache hits")
        if est > judge.max_calls:
            print(f"\n  WARNING: that exceeds the {judge.max_calls}-call budget in "
                  f"RAGAS_MAX_CALLS. The run will stop cleanly when it hits the "
                  f"ceiling rather than spend past it. Narrow the grid or raise "
                  f"the budget deliberately.")

    results = []
    print(f"\n{'chunker':<11} {'retr':<8} {'rr':<3} {'dc':<3} "
          f"{'prec':>5} {'rec':>5} {'faith':>6} {'relev':>6} {'ok':>6} {'tok/q':>7}")
    print("-" * 78)
    for chunker, mode, rr, dc in grid:
        try:
            r = run_config(docs, chunker, mode, args.top_k, rr, dc, args.invent,
                           judge=judge, workers=args.workers)
        except JudgeError as e:
            print(f"\nJUDGE ERROR after {len(results)} configurations\n{e}")
            if judge:
                judge.save_cache()
                print(f"\nCached judgements were saved, so a re-run resumes for free.")
            if not results:
                return 2
            print("Writing the report for the configurations that did complete.\n")
            break
        results.append(r)
        print(f"{chunker:<11} {mode:<8} {'Y' if rr else '-':<3} {'Y' if dc else '-':<3} "
              f"{r['context_precision']:>5.2f} {r['context_recall']:>5.2f} "
              f"{r['faithfulness']:>6.2f} {r['answer_relevancy']:>6.2f} "
              f"{r['correct']:>3}/{r['of']:<2} {r['tokens_per_query']:>7.0f}")

    write_report(results, args, judge)

    best = max(results, key=lambda r: (r["correct"], r["context_recall"], -r["tokens_per_query"]))
    print("-" * 78)
    print(f"BEST: {best['chunker']} + {best['retrieval']}"
          f"{' + rerank' if best['rerank'] else ''}"
          f"{' + decomp' if best['decompose'] else ''}  "
          f"-> {best['correct']}/{best['of']} correct, {best['tokens_per_query']:.0f} tokens/query")
    if judge is not None:
        print(f"JUDGE: {judge.stats.summary()}")
        print("       These numbers are one model's opinion. Run --variance 3 to find "
              "out how much of the gap between your top two is real.")
    print(f"\nWritten: {OUT/'ragas_report.md'}")
    print(f"         {OUT/'results.csv'}")
    print(f"         {OUT/'per_question.csv'}")
    print("\nNow open ragas_report.md and answer the five questions at the bottom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
