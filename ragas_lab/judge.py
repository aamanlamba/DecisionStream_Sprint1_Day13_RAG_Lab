"""
judge.py — the same four metrics, scored by a real model instead of by rules.

WHY THIS FILE EXISTS
--------------------
`metrics.py` computes the four RAGAS metrics deterministically. That is the
right way to LEARN what the metrics ask, because the numbers move only when
your retrieval moves.

This file is the other half of the lesson. It runs the pipeline the way it
actually runs in production:

    a real Azure OpenAI deployment writes the answer,
    and a real Azure OpenAI deployment scores it.

Both halves are needed. Run the grid in `mock` to pick a configuration, then
run the shortlist in `azure` to find out how much of your confidence survives
contact with a model that has an opinion.

WHAT CHANGES WHEN THE JUDGE IS A MODEL
--------------------------------------
1. The number is no longer reproducible. Same pipeline, same question, same
   temperature 0 — and faithfulness still moves between runs. Temperature 0
   is "as deterministic as we offer", not "deterministic".
2. The number has a cost and a latency. A 48-configuration grid is 2,880
   API calls. This module refuses to run one by accident (RAGAS_MAX_CALLS).
3. The number depends on a model VERSION you do not control. When your
   client's deployment is upgraded from gpt-4.1-mini to its successor, your
   faithfulness scores can move without a single line of your code changing.
   That is a review trigger, and it belongs in the ADR.

HOW FAITHFUL IS THIS TO THE REAL RAGAS LIBRARY?
-----------------------------------------------
Close, and the deviations are deliberate and listed. Every metric below uses
the RAGAS decomposition — the library's actual algorithm, not a vibe check.

  faithfulness        RAGAS: extract atomic statements from the answer, then
                      verdict each against the context. Score = supported/total.
                      HERE: same two steps, folded into one call that returns
                      both, to halve the cost. The decomposition is visible in
                      the JSON so you can audit it.

  context_precision   RAGAS (LLMContextPrecisionWithReference): per retrieved
                      chunk, was it useful in arriving at the reference answer?
                      Then mean average precision @ K, so a relevant chunk at
                      rank 1 counts for more than the same chunk at rank 5.
                      HERE: identical, including the MAP@K weighting.

  context_recall      RAGAS: break the REFERENCE answer into claims, mark each
                      as attributable to the retrieved context or not.
                      Score = attributable/total.  HERE: identical.

  answer_relevancy    RAGAS: generate N questions the answer could be replying
                      to, embed them, take cosine similarity with the real
                      question. Zero if the answer is noncommittal.
                      HERE: no embedding deployment is assumed, so the model
                      generates the reverse-questions AND rates each against
                      the original. Same shape, one fewer moving part. This is
                      the largest deviation in the file — say so if you quote
                      the number to a client.

  correct             Not a RAGAS metric. A separate call asks whether the
                      generated answer matches the ground truth. Without it,
                      "correct" would mean string overlap, and a real model
                      never phrases things the way your ground truth does.

Everything here is standard library. No `pip install`, no SDK — so you can
read the whole request and see exactly what is being sent.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path


# Illustrative gpt-4.1-mini rates, USD per 1M tokens. Look up the real numbers
# for your client's region and commitment before you quote anything.
PRICE_IN_PER_1M = 0.40
PRICE_OUT_PER_1M = 1.60


class JudgeError(RuntimeError):
    """Raised for anything that should stop the run: bad config, budget
    exhausted, or an Azure error we should not paper over."""


# --------------------------------------------------------------------------
# .env loading — 12 lines, no dependency.
# Real environment variables always win, so CI can override the file.
# --------------------------------------------------------------------------
def load_env(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        loaded[k] = v
        os.environ.setdefault(k, v)
    return loaded


# --------------------------------------------------------------------------
# Prompts.
#
# Each one is a separate judging task, exactly as RAGAS separates them. Note
# that every prompt forces JSON and forces a REASON. A judge that cannot say
# why it scored something is not auditable, and an evaluation you cannot audit
# is an evaluation your client cannot sign off.
# --------------------------------------------------------------------------
SYS_GENERATE = """You are the answering component of a UK motor-insurance claims assistant.

Answer ONLY from the numbered passages provided. You have no other knowledge.

Rules:
- If the passages do not contain enough to answer fully, reply with exactly:
  not_found
- Never guess a number, a clause reference, or a claim reference.
- Passages come from many different claims. Use only the ones that match the
  claim or vehicle the question names.
- Be brief: two sentences at most. Quote the figures and clause numbers you used.
"""

SYS_FAITHFULNESS = """You verify whether an answer is supported by its source passages.

Work in two steps.
STEP 1  Break the ANSWER into atomic statements. One fact per statement.
        A number, a clause reference, or a comparison is each its own statement.
STEP 2  For each statement, decide if it can be derived from the PASSAGES alone.
        General knowledge does not count. Arithmetic performed on numbers that
        ARE in the passages does count as supported.

Return JSON only:
{"statements":[{"statement":"...","supported":true,"reason":"..."}]}
"""

SYS_CONTEXT_PRECISION = """You judge whether each retrieved passage earned its place in the prompt.

For each numbered passage, decide: was this passage USEFUL in arriving at the
REFERENCE ANSWER for this question? Useful means it carries a fact the
reference answer relies on. A passage that is merely on the same topic, or is
about a different claim or a different policy version, is NOT useful.

Return JSON only, one verdict per passage, in the same order:
{"verdicts":[{"index":1,"useful":true,"reason":"..."}]}
"""

# RAGAS ships two context-precision metrics. The one above needs a reference
# answer. The negative-control questions do not have one — the correct answer
# is "this is not in the corpus" — so they use this variant, which judges
# usefulness against the QUESTION alone. Scoring them 0.0 instead would drag
# the average down by a sixth for a reason that has nothing to do with retrieval.
SYS_CONTEXT_PRECISION_NOREF = """You judge whether each retrieved passage earned its place in the prompt.

For each numbered passage, decide: could this passage help answer the QUESTION?
Helpful means it is about the specific claim, vehicle, or policy clause the
question names, and bears on what the question asks. A passage about a
different claim or a different policy version is NOT helpful, however similar
it looks.

Return JSON only, one verdict per passage, in the same order:
{"verdicts":[{"index":1,"useful":true,"reason":"..."}]}
"""

SYS_CONTEXT_RECALL = """You judge whether the retrieved passages contain what was needed.

Break the REFERENCE ANSWER into claims — one fact per claim. For each claim,
decide whether it can be attributed to the PASSAGES.

Return JSON only:
{"claims":[{"claim":"...","attributable":true,"reason":"..."}]}
"""

SYS_ANSWER_RELEVANCY = """You judge whether an answer actually addresses the question asked.

STEP 1  Read the ANSWER and write 3 questions that this answer would be a
        direct reply to.
STEP 2  Rate each generated question against the ORIGINAL question, 0.0 to 1.0,
        for how closely it asks the same thing.
STEP 3  Set noncommittal to true if the answer evades, hedges without
        committing, or says it does not know.

An answer can be perfectly true, perfectly sourced, and still score low here
because it answered a different question. That is the failure this metric exists
to catch.

Return JSON only:
{"generated":[{"question":"...","similarity":0.0}],"noncommittal":false}
"""

SYS_CORRECTNESS = """You compare a generated answer against a ground-truth answer.

Correct means: every fact the ground truth asserts is present and consistent in
the generated answer. Different wording is fine. A different NUMBER, a different
CLAUSE, or a missing half of a two-part answer is not fine.

Return JSON only:
{"correct":true,"reason":"..."}
"""


@dataclass
class JudgeStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    errors: int = 0
    latency_s: float = 0.0
    per_task: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens / 1e6 * PRICE_IN_PER_1M
                + self.completion_tokens / 1e6 * PRICE_OUT_PER_1M)

    def summary(self) -> str:
        return (f"{self.calls} API calls, {self.cache_hits} cache hits, "
                f"{self.prompt_tokens:,} in / {self.completion_tokens:,} out tokens, "
                f"~${self.cost_usd:.4f}, {self.latency_s:.0f}s of model time")


# --------------------------------------------------------------------------
# The client.
# --------------------------------------------------------------------------
class AzureJudge:
    """Generates answers and scores them with a real Azure OpenAI deployment.

    Thread-safe: the lab judges the twelve evaluation questions concurrently,
    because six sequential calls per question times twelve questions is a
    coffee break you do not need to take.
    """

    def __init__(self, endpoint: str, api_key: str, api_version: str,
                 deployment: str, temperature: float = 0.0,
                 max_calls: int = 400, cache_path: Path | None = None,
                 use_cache: bool = True, timeout: int = 90):
        missing = [n for n, v in (("AZURE_OPENAI_ENDPOINT", endpoint),
                                  ("AZURE_OPENAI_API_KEY", api_key),
                                  ("AZURE_OPENAI_DEPLOYMENT", deployment))
                   if not v]
        if missing:
            raise JudgeError(
                "Azure judge selected but these are not set: "
                + ", ".join(missing)
                + ".\nCopy .env.example to .env and fill it in, or run with --judge mock.")

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.deployment = deployment
        self.temperature = temperature
        self.max_calls = max_calls
        self.timeout = timeout
        self.use_cache = use_cache
        self.stats = JudgeStats()

        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._cache_dirty = False
        self._use_v1_route = False        # set if the classic route 404s
        if use_cache and cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except (ValueError, OSError):
                self._cache = {}

    # ---- wire format ----------------------------------------------------
    def _url(self) -> str:
        # Two routes exist and the wrong one gives you a 404 that says
        # DeploymentNotFound — which reads like a missing model and is not.
        #   classic: /openai/deployments/<deployment>/chat/completions?api-version=...
        #   v1:      /openai/v1/chat/completions   with "model" in the body
        if self._use_v1_route:
            return f"{self.endpoint}/openai/v1/chat/completions"
        return (f"{self.endpoint}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}")

    def _post(self, body: dict) -> dict:
        payload = dict(body)
        if self._use_v1_route:
            payload["model"] = self.deployment
        req = urllib.request.Request(
            self._url(),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "api-key": self.api_key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _chat(self, system: str, user: str, task: str,
              max_tokens: int = 700, json_mode: bool = True) -> str:
        """One judged call, with cache, budget, and retries.

        The cache key covers the prompt AND the deployment AND the temperature.
        Change any of them and you are asking a different judge a different
        question, so the old answer is not reusable.
        """
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        key = sha256(json.dumps(
            [self.deployment, self.api_version, task, body],
            sort_keys=True).encode()).hexdigest()

        if self.use_cache:
            with self._lock:
                hit = self._cache.get(key)
            if hit is not None:
                with self._lock:
                    self.stats.cache_hits += 1
                return hit

        with self._lock:
            if self.stats.calls >= self.max_calls:
                raise JudgeError(
                    f"Call budget exhausted ({self.max_calls}). This is the guard that "
                    f"stops a 48-configuration grid from becoming a 2,880-call invoice.\n"
                    f"Raise RAGAS_MAX_CALLS in .env, or narrow the grid.")
            self.stats.calls += 1

        last: Exception | None = None
        for attempt in range(4):
            t0 = time.time()
            try:
                data = self._post(body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:400]
                # Wrong route, not a missing deployment. Switch once and retry.
                if e.code == 404 and not self._use_v1_route:
                    self._use_v1_route = True
                    last = e
                    continue
                if e.code in (408, 409, 429, 500, 502, 503, 504):
                    wait = float(e.headers.get("retry-after", 0) or (2 ** attempt))
                    with self._lock:
                        self.stats.retries += 1
                    time.sleep(min(wait, 30))
                    last = e
                    continue
                with self._lock:
                    self.stats.errors += 1
                raise JudgeError(f"Azure returned HTTP {e.code} for task '{task}': {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                with self._lock:
                    self.stats.retries += 1
                time.sleep(2 ** attempt)
                last = e
                continue

            usage = data.get("usage") or {}
            choice = (data.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            with self._lock:
                self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
                self.stats.completion_tokens += usage.get("completion_tokens", 0)
                self.stats.latency_s += time.time() - t0
                self.stats.per_task[task] = self.stats.per_task.get(task, 0) + 1
                if self.use_cache:
                    self._cache[key] = text
                    self._cache_dirty = True
            if choice.get("finish_reason") == "content_filter":
                raise JudgeError(
                    f"Azure content filter blocked task '{task}'. The claims corpus is "
                    f"synthetic; if this fires, check the filter policy on the deployment.")
            return text

        with self._lock:
            self.stats.errors += 1
        raise JudgeError(f"Azure call failed after retries for task '{task}': {last}")

    def save_cache(self) -> None:
        if self.use_cache and self._cache_path and self._cache_dirty:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=0))
            self._cache_dirty = False

    # ---- parsing --------------------------------------------------------
    @staticmethod
    def _json(text: str, task: str) -> dict:
        try:
            return json.loads(text)
        except ValueError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except ValueError:
                    pass
        raise JudgeError(f"Judge returned non-JSON for task '{task}': {text[:200]}")

    @staticmethod
    def _passages(chunks) -> str:
        return "\n\n".join(
            f"[{i + 1}] (source: {c.doc}, corpus: {c.corpus})\n{c.text}"
            for i, c in enumerate(chunks))

    # ---- generation -----------------------------------------------------
    def generate_answer(self, question: str, chunks) -> str:
        if not chunks:
            return "not_found"
        user = f"PASSAGES:\n{self._passages(chunks)}\n\nQUESTION: {question}"
        out = self._chat(SYS_GENERATE, user, "generate", max_tokens=250, json_mode=False)
        return out or "not_found"

    @staticmethod
    def refused(answer: str) -> bool:
        """Did the model decline to answer?

        The prompt asks for exactly `not_found`. The model complies about half
        the time and otherwise writes a paragraph that ENDS in not_found, or
        refuses in plain English without the token at all. If you only check
        `startswith`, those refusals get scored as answers — and a refusal
        scored as an answer is a false failure on every generation-side metric.

        This is the argument for structured outputs in production. A refusal
        you cannot detect is a refusal you cannot measure.
        """
        a = answer.strip().lower()
        if "not_found" in a or a.startswith("not found"):
            return True
        return bool(re.search(
            r"(do(es)? not|don't|cannot|can't|unable to)\s+(contain|state|specify|provide|find|answer)"
            r"|is not (stated|specified|provided|mentioned|available) in the (passages|context|documents)"
            r"|no information (is )?(available|provided|found)",
            a))

    # ---- the four metrics ----------------------------------------------
    def faithfulness(self, answer: str, chunks) -> tuple[float, list]:
        # A refusal asserts nothing, so there is nothing to be unfaithful about.
        if self.refused(answer):
            return 1.0, []
        user = (f"PASSAGES:\n{self._passages(chunks)}\n\nANSWER: {answer}")
        d = self._json(self._chat(SYS_FAITHFULNESS, user, "faithfulness"), "faithfulness")
        st = d.get("statements") or []
        if not st:
            return 1.0, []
        return sum(1 for s in st if s.get("supported")) / len(st), st

    def context_precision(self, question: str, reference: str | None, chunks) -> tuple[float, list]:
        """Mean average precision @ K — rank-aware, exactly as RAGAS does it.

        Two configurations can retrieve the same relevant chunk and score
        differently, because one put it at rank 1 and the other at rank 5.
        That difference is real: everything above it is tokens the model reads
        first and you paid for.

        `reference=None` selects the without-reference variant, for the
        negative-control questions that have no correct answer to compare to.
        """
        if not chunks:
            return 0.0, []
        if reference is None:
            sys_prompt, task = SYS_CONTEXT_PRECISION_NOREF, "context_precision_noref"
            user = f"QUESTION: {question}\n\nPASSAGES:\n{self._passages(chunks)}"
        else:
            sys_prompt, task = SYS_CONTEXT_PRECISION, "context_precision"
            user = (f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\n\n"
                    f"PASSAGES:\n{self._passages(chunks)}")
        d = self._json(self._chat(sys_prompt, user, task), task)
        v = d.get("verdicts") or []
        rel = [bool(x.get("useful")) for x in v][:len(chunks)]
        rel += [False] * (len(chunks) - len(rel))
        total_rel = sum(rel)
        if total_rel == 0:
            return 0.0, v
        run, ap = 0, 0.0
        for k, is_rel in enumerate(rel, start=1):
            if is_rel:
                run += 1
                ap += run / k
        return ap / total_rel, v

    def context_recall(self, reference: str, chunks) -> tuple[float, list]:
        user = (f"PASSAGES:\n{self._passages(chunks)}\n\nREFERENCE ANSWER: {reference}")
        d = self._json(self._chat(SYS_CONTEXT_RECALL, user, "context_recall"), "context_recall")
        cl = d.get("claims") or []
        if not cl:
            return 0.0, []
        return sum(1 for c in cl if c.get("attributable")) / len(cl), cl

    def answer_relevancy(self, question: str, answer: str, case: dict) -> tuple[float, dict]:
        # A refusal to an unanswerable question is exactly right. A refusal to
        # an answerable one is honest and useless — and the score must say so,
        # or a system that refuses everything looks perfect.
        if self.refused(answer):
            return (1.0 if case.get("expect_no_answer") else 0.3), {"refusal": True}
        user = f"ORIGINAL QUESTION: {question}\n\nANSWER: {answer}"
        d = self._json(self._chat(SYS_ANSWER_RELEVANCY, user, "answer_relevancy"),
                       "answer_relevancy")
        if d.get("noncommittal"):
            return 0.0, d
        sims = [float(g.get("similarity", 0.0)) for g in (d.get("generated") or [])]
        if not sims:
            return 0.0, d
        return max(0.0, min(1.0, sum(sims) / len(sims))), d

    def answer_correct(self, question: str, answer: str, truth: str) -> tuple[bool, str]:
        user = (f"QUESTION: {question}\nGROUND TRUTH: {truth}\nGENERATED ANSWER: {answer}")
        d = self._json(self._chat(SYS_CORRECTNESS, user, "correctness", max_tokens=300),
                       "correctness")
        return bool(d.get("correct")), str(d.get("reason", ""))[:200]

    # ---- one evaluation case -------------------------------------------
    def evaluate_case(self, chunks, case: dict):
        """Same signature as metrics.evaluate_case, so run_lab.py does not care
        which judge it is holding."""
        from metrics import Scores        # local import: keeps metrics.py judge-free

        question = case["question"]
        reference = case["answer"]
        answer = self.generate_answer(question, chunks)
        answered = not self.refused(answer)

        fa, _ = self.faithfulness(answer, chunks)
        ar, _ = self.answer_relevancy(question, answer, case)

        if case.get("expect_no_answer"):
            # No reference answer exists, so context RECALL has nothing to
            # measure — there are no required claims to be missing. It is
            # vacuously satisfied, which is what the mock judge reports too.
            # Context PRECISION still means something: did we retrieve passages
            # that bear on the question at all? So we use RAGAS's
            # without-reference variant rather than scoring it zero.
            cp, _ = self.context_precision(question, None, chunks)
            cr = 1.0
            correct = not answered
        else:
            cp, _ = self.context_precision(question, reference, chunks)
            cr, _ = self.context_recall(reference, chunks)
            correct = answered and self.answer_correct(question, answer, reference)[0]

        return Scores(cp, cr, fa, ar, answered, correct), answer


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_judge(mode: str, here: Path, use_cache: bool = True,
                temperature: float | None = None):
    """Returns None for mock mode — run_lab.py then uses metrics.evaluate_case."""
    if mode == "mock":
        return None
    if mode != "azure":
        raise JudgeError(f"unknown judge mode: {mode!r} (expected 'mock' or 'azure')")
    return AzureJudge(
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        temperature=(temperature if temperature is not None
                     else float(os.environ.get("RAGAS_JUDGE_TEMPERATURE", "0.0"))),
        max_calls=int(os.environ.get("RAGAS_MAX_CALLS", "400")),
        cache_path=here / "out" / "judge_cache.json",
        use_cache=use_cache,
    )
