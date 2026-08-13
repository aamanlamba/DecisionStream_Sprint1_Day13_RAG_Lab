"""
metrics.py — the four RAGAS metrics, implemented from first principles.

WHY WE ARE NOT JUST CALLING THE RAGAS LIBRARY
---------------------------------------------
Two reasons, and the second is the important one.

1. The library needs a live model and an API key. This lab runs offline so
   that everyone in the pod can complete it.

2. RAGAS metrics are normally computed BY AN LLM ACTING AS A JUDGE. That
   means your evaluation numbers are themselves model outputs — with all the
   variance, cost and version-sensitivity that implies. A team that treats
   "faithfulness 0.87" as a measurement, rather than as one model's opinion
   with a confidence interval, will over-trust it.

   Implementing the metrics with deterministic rules makes what they are
   actually asking completely visible. When you move to the real library in
   production, you will know exactly what it is approximating — and that a
   change in judge model version can move your scores without your pipeline
   changing at all.

THE FOUR METRICS, IN PLAIN WORDS
--------------------------------
  CONTEXT PRECISION   Of the passages we retrieved, how many were relevant?
                      Low = we are stuffing the prompt with noise.

  CONTEXT RECALL      Of the passages needed to answer, how many did we get?
                      Low = the answer was never available to the model.

  FAITHFULNESS        Is every claim in the answer supported by the retrieved
                      passages? Low = the model is drawing on something else,
                      which in production means inventing.

  ANSWER RELEVANCY    Does the answer address the question that was asked?
                      Low = fluent, sourced, and about something else.

The interesting failures are the ASYMMETRIC ones:
  high faithfulness + low relevancy  -> correctly citing the wrong thing
  high recall + low precision        -> right answer buried in noise
  high precision + low recall        -> confident answer, missing half the facts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from retrieval import tokenize, Chunk


@dataclass
class Scores:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    answered: bool
    correct: bool

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _norm(s: str) -> str:
    return re.sub(r"[\s,]+", "", s.lower())


def _contains(text: str, needle: str) -> bool:
    return _norm(needle) in _norm(text)


# --------------------------------------------------------------------------
# Retrieval-side metrics
# --------------------------------------------------------------------------
def context_recall(chunks: list[Chunk], must_contain: list[str]) -> float:
    """Of the facts required to answer, how many are present in what we
    retrieved? If the answer was never in the context, nothing downstream
    can fix it."""
    if not must_contain:
        return 1.0
    blob = " ".join(c.text for c in chunks)
    hits = sum(1 for m in must_contain if _contains(blob, m))
    return hits / len(must_contain)


def context_precision(chunks: list[Chunk], must_contain: list[str],
                      question: str) -> float:
    """Of the passages we retrieved, how many actually earn their place?

    A passage is relevant if it carries one of the required facts, or has
    meaningful overlap with the question. Everything else is noise the model
    has to read past — and that you paid tokens for.
    """
    if not chunks:
        return 0.0
    q = set(tokenize(question))
    rel = 0
    for c in chunks:
        if must_contain and any(_contains(c.text, m) for m in must_contain):
            rel += 1
            continue
        overlap = len(q & set(tokenize(c.text))) / max(1, len(q))
        if overlap >= 0.35:
            rel += 1
    return rel / len(chunks)


# --------------------------------------------------------------------------
# Generation-side metrics
#
# We simulate the generator deterministically: it answers from the retrieved
# context if the facts are there, and otherwise either refuses (good) or
# invents (bad, and configurable so you can see the effect).
# --------------------------------------------------------------------------
def simulate_answer(chunks: list[Chunk], case: dict,
                    invents_when_missing: bool = False) -> tuple[str, bool]:
    """Returns (answer_text, invented_flag).

    This stands in for the LLM. Keeping it deterministic means the numbers
    you see change ONLY when your retrieval changes — which is the whole
    point of a retrieval experiment. Introduce a real model and you can no
    longer tell whether a score moved because of your chunk size or because
    of model variance.
    """
    blob = " ".join(c.text for c in chunks)
    need = case.get("must_contain", [])
    have = [m for m in need if _contains(blob, m)]

    if case.get("expect_no_answer"):
        # The honest behaviour is to say it is not there.
        if invents_when_missing:
            return ("Based on the documents, liability appears to have been "
                    "admitted by the third party.", True)
        return ("not_found — this is not present in the retrieved passages.", False)

    if need and len(have) < len(need):
        if invents_when_missing:
            return (f"{case['answer']} (stated without full support in context)", True)
        return ("not_found — the retrieved passages do not contain enough to "
                "answer this fully.", False)

    return (case["answer"], False)


def faithfulness(answer: str, chunks: list[Chunk], invented: bool) -> float:
    """Is every claim in the answer supported by what we retrieved?

    Approximated by checking that the answer's content-bearing tokens —
    especially numbers and clause references — appear in the context.
    """
    if invented:
        return 0.0
    if answer.startswith("not_found"):
        return 1.0        # refusing IS faithful; it claims nothing
    blob_toks = set()
    for c in chunks:
        blob_toks |= set(tokenize(c.text))
    a_toks = [t for t in tokenize(answer)]
    facts = [t for t in a_toks if re.search(r"\d", t)] or a_toks
    if not facts:
        return 1.0
    return sum(1 for t in facts if t in blob_toks) / len(facts)


def answer_relevancy(answer: str, question: str, case: dict) -> float:
    """Does the answer address the question asked?

    A refusal to a question that HAS an answer is honest but not relevant —
    and the scorecard should say so, because a system that refuses everything
    is faithful and useless.
    """
    if case.get("expect_no_answer"):
        return 1.0 if answer.startswith("not_found") else 0.0
    if answer.startswith("not_found"):
        return 0.3        # honest, but it did not answer
    q = set(tokenize(question))
    a = set(tokenize(answer))
    if not q:
        return 0.0
    overlap = len(q & a) / len(q)
    return min(1.0, 0.45 + overlap)


def evaluate_case(chunks: list[Chunk], case: dict,
                  invents_when_missing: bool = False) -> tuple[Scores, str]:
    answer, invented = simulate_answer(chunks, case, invents_when_missing)
    cp = context_precision(chunks, case.get("must_contain", []), case["question"])
    cr = context_recall(chunks, case.get("must_contain", []))
    fa = faithfulness(answer, chunks, invented)
    ar = answer_relevancy(answer, case["question"], case)
    answered = not answer.startswith("not_found")

    if case.get("expect_no_answer"):
        correct = not answered
    else:
        correct = answered and not invented and cr >= 0.999

    return Scores(cp, cr, fa, ar, answered, correct), answer


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
