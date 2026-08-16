# RAGAS evaluation report — DecisionStream AI retrieval

Configurations evaluated: **48**   |   Questions: **12**   |   top_k: **4**

**Judge: `mock`** — metrics computed deterministically in `ragas_lab/metrics.py`. These numbers are reproducible to the digit: re-run this and nothing moves. That is a property of the harness, not of your pipeline.

Re-run the shortlist with `--judge azure` before anything here reaches an ADR. A real judge scores generated answers, not simulated ones, and it disagrees.

## All configurations

| Chunker | Retrieval | Rerank | Decomp | Prec | Recall | Faith | Relev | Correct | Tokens/q | $/1k q |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_512 | keyword | - | - | 0.65 | 0.96 | 0.84 | 0.66 | 11/12 | 701 | $0.105 |
| fixed_1024 | keyword | - | - | 0.65 | 0.96 | 0.84 | 0.66 | 11/12 | 701 | $0.105 |
| fixed_512 | keyword | - | Y | 0.65 | 0.94 | 0.91 | 0.62 | 10/12 | 701 | $0.105 |
| fixed_1024 | keyword | - | Y | 0.65 | 0.94 | 0.91 | 0.62 | 10/12 | 701 | $0.105 |
| fixed_256 | keyword | - | - | 0.65 | 0.91 | 0.96 | 0.58 | 9/12 | 541 | $0.081 |
| fixed_256 | keyword | - | Y | 0.65 | 0.89 | 0.96 | 0.58 | 9/12 | 543 | $0.082 |
| structure | keyword | - | - | 0.58 | 0.89 | 0.95 | 0.58 | 9/12 | 277 | $0.042 |
| structure | keyword | - | Y | 0.60 | 0.89 | 0.95 | 0.58 | 9/12 | 273 | $0.041 |
| structure | hybrid | - | - | 0.65 | 0.89 | 0.95 | 0.58 | 9/12 | 256 | $0.038 |
| fixed_256 | hybrid | - | - | 0.67 | 0.88 | 0.96 | 0.58 | 9/12 | 488 | $0.073 |
| fixed_256 | hybrid | Y | - | 0.65 | 0.88 | 0.96 | 0.58 | 9/12 | 465 | $0.070 |
| fixed_512 | vector | - | - | 0.60 | 0.88 | 0.96 | 0.58 | 9/12 | 485 | $0.073 |
| fixed_512 | vector | Y | - | 0.60 | 0.88 | 0.97 | 0.58 | 9/12 | 514 | $0.077 |
| fixed_512 | hybrid | Y | - | 0.62 | 0.88 | 0.97 | 0.58 | 9/12 | 561 | $0.084 |
| fixed_1024 | vector | - | - | 0.60 | 0.88 | 0.96 | 0.58 | 9/12 | 485 | $0.073 |
| fixed_1024 | vector | Y | - | 0.60 | 0.88 | 0.97 | 0.58 | 9/12 | 514 | $0.077 |
| fixed_1024 | hybrid | Y | - | 0.62 | 0.88 | 0.97 | 0.58 | 9/12 | 561 | $0.084 |
| fixed_512 | vector | Y | Y | 0.60 | 0.86 | 0.97 | 0.58 | 9/12 | 516 | $0.077 |
| fixed_512 | hybrid | Y | Y | 0.62 | 0.86 | 0.97 | 0.58 | 9/12 | 564 | $0.085 |
| fixed_1024 | vector | Y | Y | 0.60 | 0.86 | 0.97 | 0.58 | 9/12 | 516 | $0.077 |
| fixed_1024 | hybrid | Y | Y | 0.62 | 0.86 | 0.97 | 0.58 | 9/12 | 564 | $0.085 |
| structure | hybrid | Y | - | 0.62 | 0.86 | 0.95 | 0.58 | 9/12 | 251 | $0.038 |
| fixed_256 | vector | - | - | 0.65 | 0.85 | 0.96 | 0.58 | 9/12 | 439 | $0.066 |
| fixed_256 | vector | Y | - | 0.60 | 0.85 | 0.96 | 0.58 | 9/12 | 431 | $0.065 |
| fixed_256 | vector | Y | Y | 0.58 | 0.83 | 0.96 | 0.58 | 9/12 | 401 | $0.060 |
| fixed_256 | hybrid | - | Y | 0.62 | 0.83 | 0.96 | 0.58 | 9/12 | 455 | $0.068 |
| fixed_256 | hybrid | Y | Y | 0.62 | 0.83 | 0.96 | 0.58 | 9/12 | 448 | $0.067 |
| fixed_512 | vector | - | Y | 0.58 | 0.83 | 0.96 | 0.58 | 9/12 | 453 | $0.068 |
| fixed_1024 | vector | - | Y | 0.58 | 0.83 | 0.96 | 0.58 | 9/12 | 453 | $0.068 |
| structure | vector | - | - | 0.62 | 0.83 | 0.95 | 0.58 | 9/12 | 243 | $0.036 |
| structure | vector | Y | - | 0.56 | 0.83 | 0.95 | 0.58 | 9/12 | 225 | $0.034 |
| structure | vector | - | Y | 0.65 | 0.83 | 0.95 | 0.58 | 9/12 | 239 | $0.036 |
| structure | vector | Y | Y | 0.58 | 0.83 | 0.95 | 0.58 | 9/12 | 221 | $0.033 |
| structure | hybrid | - | Y | 0.62 | 0.83 | 0.95 | 0.58 | 9/12 | 248 | $0.037 |
| structure | hybrid | Y | Y | 0.62 | 0.83 | 0.95 | 0.58 | 9/12 | 242 | $0.036 |
| fixed_256 | keyword | Y | - | 0.67 | 0.87 | 0.96 | 0.57 | 8/12 | 525 | $0.079 |
| fixed_512 | keyword | Y | - | 0.62 | 0.87 | 0.97 | 0.57 | 8/12 | 642 | $0.096 |
| fixed_512 | keyword | Y | Y | 0.65 | 0.87 | 0.97 | 0.57 | 8/12 | 645 | $0.097 |
| fixed_1024 | keyword | Y | - | 0.62 | 0.87 | 0.97 | 0.57 | 8/12 | 642 | $0.096 |
| fixed_1024 | keyword | Y | Y | 0.65 | 0.87 | 0.97 | 0.57 | 8/12 | 645 | $0.097 |
| fixed_256 | keyword | Y | Y | 0.67 | 0.85 | 0.96 | 0.57 | 8/12 | 528 | $0.079 |
| structure | keyword | Y | - | 0.62 | 0.85 | 0.95 | 0.57 | 8/12 | 267 | $0.040 |
| structure | keyword | Y | Y | 0.65 | 0.85 | 0.95 | 0.57 | 8/12 | 263 | $0.040 |
| fixed_512 | hybrid | - | - | 0.62 | 0.84 | 0.97 | 0.57 | 8/12 | 590 | $0.089 |
| fixed_1024 | hybrid | - | - | 0.62 | 0.84 | 0.97 | 0.57 | 8/12 | 590 | $0.089 |
| fixed_256 | vector | - | Y | 0.62 | 0.79 | 1.00 | 0.52 | 8/12 | 400 | $0.060 |
| fixed_512 | hybrid | - | Y | 0.60 | 0.79 | 0.97 | 0.57 | 8/12 | 558 | $0.084 |
| fixed_1024 | hybrid | - | Y | 0.60 | 0.79 | 0.97 | 0.57 | 8/12 | 558 | $0.084 |

## Where each configuration wins and loses

Averages hide the interesting part. This table is correctness by question type.

| Chunker | Retrieval | Rerank | Decomp | clause_lookup | multi_part | negative | single_fact | version_sensitive |
|---|---|---|---|---|---|---|---|---|
| fixed_512 | keyword | - | - | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | keyword | - | - | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 |
| fixed_512 | keyword | - | Y | 1.00 | 0.67 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | keyword | - | Y | 1.00 | 0.67 | 1.00 | 0.67 | 1.00 |
| fixed_256 | keyword | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | keyword | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | vector | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | vector | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | vector | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | hybrid | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | hybrid | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | hybrid | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | hybrid | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_512 | vector | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_512 | vector | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_512 | vector | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_512 | vector | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_512 | hybrid | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_512 | hybrid | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | vector | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | vector | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | vector | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | vector | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | hybrid | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_1024 | hybrid | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | keyword | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | keyword | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | vector | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | vector | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | vector | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | vector | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | hybrid | - | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | hybrid | Y | - | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | hybrid | - | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| structure | hybrid | Y | Y | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| fixed_256 | keyword | Y | - | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_256 | keyword | Y | Y | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_256 | vector | - | Y | 1.00 | 0.00 | 1.00 | 0.67 | 1.00 |
| fixed_512 | keyword | Y | - | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_512 | keyword | Y | Y | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_512 | hybrid | - | - | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_512 | hybrid | - | Y | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_1024 | keyword | Y | - | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_1024 | keyword | Y | Y | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_1024 | hybrid | - | - | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| fixed_1024 | hybrid | - | Y | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| structure | keyword | Y | - | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |
| structure | keyword | Y | Y | 1.00 | 0.33 | 1.00 | 0.33 | 1.00 |

## Why 512 and 1024 give identical results here

Both produce **37 chunks**. Most documents in this corpus are shorter than 512 tokens, so a 512-token and a 1024-token chunker never actually split them differently.

This is not a bug and it is worth understanding: **chunk size stops being a variable once it exceeds your typical document length.** Tuning it further is effort spent on a parameter that has no effect on your data. Check your document length distribution before you run a chunk size experiment — if most documents fit in one chunk, the experiment has already answered itself.


## Best configuration on this evaluation set

**fixed_512 + keyword** — 11/12 correct, recall 0.96, precision 0.65, 701 tokens per query.

## Questions to answer before this goes in the ADR

1. Which question TYPE does your chosen configuration handle worst, and what happens in production when that type arrives?
2. Recall and precision move in opposite directions as chunk size grows. Which one does your client's risk appetite favour, and who agreed that?
3. What does the cost column look like at your real query volume, not at twelve questions?
4. The version-sensitive questions (Q09, Q10) exist because the policy corpus contains two versions of the same clause. Did your configuration retrieve the right one — and would you have noticed if it had not?
5. What is your review trigger for re-running this evaluation?
