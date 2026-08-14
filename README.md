# Sprint 1 · Day 13 Lab — RAG Evaluation & Optimisation

**A teaching lab on evaluating retrieval — and on not over-trusting the evaluation.**

Four chunking strategies × three retrieval modes × re-ranking × query
decomposition, scored on four RAGAS metrics across twelve questions.
Test RAGAS
---

## It runs offline

```bash
python3 run_lab.py --top-k 4
```

Standard library only. No Azure, no API key, no `pip install`. If you have
Python 3.10+, that works right now.

---

## Two judges, and you need both

| | `mock` (default) | `azure` |
|---|---|---|
| Who writes the answer | a deterministic stand-in | **a real Azure OpenAI deployment** |
| Who scores it | rules in `ragas_lab/metrics.py` | **the same deployment, RAGAS-style** |
| Cost / time for the grid | free, seconds | ~$0.06, ~2 min for the shortlist |
| Same numbers twice? | **yes, to the digit** | **no** — and that is the lesson |
| What it is for | screening 48 configurations | evaluating the 4 you shortlisted |

```bash
cp .env.example .env          # then fill in your endpoint, key, deployment
python3 run_lab.py --judge azure              # the 4-config shortlist
python3 run_lab.py --judge azure --variance 3 # the noise floor — run this first
python3 run_lab.py --judge azure --full       # all 48. ~12x the cost.
```

`RAGAS_JUDGE` in `.env` sets the default; `--judge` overrides it. `.env` is
git-ignored — the key never goes in the repo.

Answers and judgements are cached in `out/judge_cache.json`, so a re-run costs
almost nothing and the run stops cleanly at `RAGAS_MAX_CALLS` rather than
quietly spending past it.

**Read `Interpreting_RAGAS_Results.docx` once you have output.** It works
through seven real rows from an azure run and the three ways the aggregate
table will mislead you.

---

## Why we also implement the metrics by hand

The `mock` judge is not a lesser version of the `azure` one. It is a different
instrument, and the pair is the point.

**1.** Everyone in the pod can complete the lab without a key, a quota, or a
bill.

**2.** RAGAS metrics are normally computed **by an LLM acting as a judge**.
Your evaluation numbers are therefore themselves model outputs — with variance,
cost, and sensitivity to the judge's version. A team that reads
"faithfulness 0.87" as a measurement rather than as one model's opinion will
over-trust it.

Implementing them deterministically makes what they actually ask completely
visible — and then running the same configurations through a real judge shows
you exactly how much the approximation cost you. On this corpus the two judges
**rank the configurations differently**, which is the most useful thing the lab
produces.

A judge model upgrade can move your scores while your pipeline has not changed
at all. **That is a review trigger**, and `ragas_lab/judge.py` is where you can
see why.

---

## The four metrics, in plain words

| Metric | The question it asks | What a low score means |
|---|---|---|
| **Context precision** | Of the passages we retrieved, how many were relevant? | You are paying tokens to send the model noise |
| **Context recall** | Of the passages needed, how many did we get? | The answer was never available. Nothing downstream can fix this |
| **Faithfulness** | Is every claim in the answer supported by the context? | The model is drawing on something else — in production, inventing |
| **Answer relevancy** | Does the answer address the question asked? | Fluent, sourced, and about something else |

The interesting failures are **asymmetric**:

- high faithfulness + low relevancy → correctly citing the wrong thing
- high recall + low precision → right answer buried in noise you paid for
- high precision + low recall → confident answer, missing half the facts

---

## What to do

### 1 · Run the grid (10 min)

```bash
python3 run_lab.py --top-k 4
```

48 configurations. Read `out/ragas_report.md`.

### 2 · Look past the averages (15 min)

The report's second table is **correctness by question type**. This is where
the decision actually lives. A configuration that scores well on average but
fails every multi-part question is not "slightly worse" — it is the wrong
shape for the work.

The five question types:

| Type | What it tests |
|---|---|
| `single_fact` | One value, from one document, among 12 similar claims |
| `clause_lookup` | Finding a numbered clause — keyword's home ground |
| `multi_part` | Two questions in one sentence — where decomposition should help |
| `version_sensitive` | Two versions of the same clause exist. Did you get the right one? |
| `negative` | The answer is not in the corpus. Does it say so? |

### 3 · Inspect one question closely (10 min)

```bash
python3 run_lab.py --show Q06     # multi-part
python3 run_lab.py --show Q09     # version-sensitive — read the TRAP line
python3 run_lab.py --show Q11     # negative
```

### 4 · See what happens when the model guesses (5 min)

```bash
python3 run_lab.py --top-k 4 --invent
```

The simulated generator now fills gaps instead of refusing. Watch faithfulness
fall while relevancy *rises*. That combination — a system that looks more
helpful and is less trustworthy — is exactly what an evaluation set exists to
catch, and exactly what a demo will not show you.

### 5 · Put a real model behind it (20 min)

```bash
python3 run_lab.py --judge azure --variance 3   # first: how noisy is the judge?
python3 run_lab.py --judge azure                # then: the shortlist
```

Open `out/per_question.csv` and read the **`answer` column**. That is the whole
exercise. On this corpus you will find an answer that scores faithfulness 1.00
and quotes a figure belonging to a different claim, and another that scores
1.00 on all four metrics and is still unusable. Work out why the metrics did
not catch them before you read the explanations in
`Interpreting_RAGAS_Results.docx`.

### 6 · Answer the five questions at the bottom of the report (15 min)

They go into the ADR. The doc's Section 7 has two more.

---

## What you should find

Some of these will surprise people. Let them find them rather than telling them.

- **Keyword beats vector on clause lookups.** A clause number is an identifier,
  not a meaning. Semantic search is blind to it.
- **Vector beats keyword on descriptive questions**, where the wording of the
  question and the wording of the document differ.
- **Structure-aware chunking uses roughly half the tokens** of fixed chunking
  for comparable correctness. At scale that is the cost line.
- **512 and 1024 give identical results** on this corpus. The report explains
  why, and it is not a bug — it is a finding about your data.
- **Re-ranking helps some configurations and hurts others.** Retrieve-wide-then-
  narrow only pays when the wide set actually contained something better.
- **Decomposition helps multi-part questions and costs you elsewhere**, because
  merging two candidate sets dilutes precision.

---

## Files

```
run_lab.py                     the command you run
.env / .env.example            judge selection + Azure OpenAI config (.env is git-ignored)
Interpreting_RAGAS_Results.docx  how to read the output without fooling yourself
ragas_lab/
  corpus.py                    2 policy versions, 1 real claim, 10 distractor claims,
                               and the 12-question evaluation set
  retrieval.py                 chunkers, keyword/vector/hybrid, re-ranker, decomposition
  metrics.py                   the four RAGAS metrics from first principles (mock judge)
  judge.py                     the same four, scored by a real Azure OpenAI deployment
out/                           ragas_report.md · results.csv · per_question.csv
                               judge_variance.md · judge_cache.json
```

---

## Three honest limitations

**The "vector" retriever is TF-IDF cosine, not learned embeddings.** It behaves
like a vector search for this lab's purposes — rewards meaning overlap, blind to
exact identifiers — but it will not capture synonymy the way a real embedding
model does. Where a real embedding model would do better is on questions whose
wording shares no vocabulary with the source.

**In `mock` mode the generator is simulated deterministically.** That is on
purpose: a score only moves when *your retrieval* changes. It also means mock's
`correct` column is optimistic — it checks whether the required strings appear
anywhere in the joined context, so a figure retrieved from the wrong claim
still counts. `--judge azure` is what exposes that; the two judges disagree by
three questions out of twelve on this corpus.

**`answer_relevancy` under the azure judge is the one real deviation from the
RAGAS algorithm.** RAGAS generates reverse-questions and compares them to the
original with an *embedding* model. This lab assumes no embedding deployment,
so the judge generates the reverse-questions and rates them itself. Same shape,
one fewer moving part — say so if you quote the number to a client. The other
three metrics use the library's actual decomposition, MAP@K weighting included.

---

## Moving this to Azure

The mechanics you have just read are what Azure AI Search does for you:

| In this lab | In Azure AI Search |
|---|---|
| `chunk_fixed` / `chunk_structure` | Index projections, or your own skillset |
| `keyword_scores` | BM25 full-text search |
| `vector_scores` | Vector search over an embedding field |
| `hybrid` | Hybrid search — one query, both scores fused |
| `rerank` | Semantic ranker |
| `decompose` | Your orchestration layer — not a search feature |

For the **Azure Monitor custom metric** in today's brief, emit
`context_recall` and `context_precision` per query from your pipeline. Recall is
the one to alert on: it is the metric that tells you the answer was never
available, and it is the failure a user cannot see.

**Verify current service names and API versions before you build.** Azure has
renamed and restructured search and Foundry more than once, and checking is the
habit, not the inconvenience.

---

## Before you present

- [ ] You can name which question type your chosen configuration handles worst
- [ ] You can explain why keyword beat vector somewhere, in one sentence
- [ ] You ran `--invent` and can describe what happened to the two metrics
- [ ] You have a token-cost figure at your real query volume, not at 12 questions
- [ ] You have a review trigger for re-running this evaluation
- [ ] You measured the judge's noise floor and no gap you quote is smaller than it
- [ ] You can name the judge deployment, version and temperature behind your numbers
- [ ] You have read at least three raw answers, not just the scorecard
