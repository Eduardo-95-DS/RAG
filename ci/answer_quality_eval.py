#!/usr/bin/env python3
"""
Answer-quality evaluation gate for Cloud Build (manual trigger).

Complements ci/retrieval_eval.py, which only checks whether the *retriever*
finds the right chunks. This script checks the full pipeline's *output*:
runs the real rewriter -> responder -> guardrail graph (GraphBuilder.run,
same code path streamlit_app.py uses) against a fixed question set, then
scores the final, user-facing answers with the Vertex AI Gen AI evaluation
service (LLM-as-judge).

Metrics used (both reference-free — no golden answers required). Note the
two metrics are on DIFFERENT scales — confirmed against Vertex's predefined
metric docs after the first real run, don't assume both are 0-1:
  groundedness               - 0-1 scale (binary per example: grounded or
                               not; the mean is a fraction). Is the answer
                               supported by the retrieved context? Directly
                               mirrors what ground_check in reactnode.py
                               already tries to enforce, but scored by an
                               independent judge model instead of the app's
                               own Groq call.
  question_answering_quality - 1-5 scale (rating rubric, 5=best). Is this a
                               good, well-formed answer overall? Broader
                               than groundedness; ground_check never checks
                               this at all. First real run: 4.76/5 mean.

Reference-based correctness (item 6): golden answers for all 25 questions
are sourced from the actual PDF (see QA_PAIRS). Vertex's built-in
question_answering_correctness was REMOVED from the Gen AI eval service
(confirmed 2026-07-15: "Metric name: ... is not supported"), and the
surviving reference-based built-ins (BLEU/ROUGE/exact_match) measure lexical
overlap, not factual correctness ("$130.5 billion" vs golden "$130,497
million" would score as wrong). So correctness is computed LOCALLY by
key_figure_correctness(): fraction of answers containing an accepted
PDF-verified figure/term for their question — deterministic, no API call,
0-1 scale, and robust to phrasing. Analogous to unit-tests-as-eval for
answers that have exact expected values.

Usage
-----
    python ci/answer_quality_eval.py --fail-under-groundedness=0.7 \\
        --fail-under-qa-quality=3.5

Requires faiss_index/ to exist (pulled from GCS by the calling Cloud Build
step) and GROQ_API_KEY set in the environment (the app's own LLM, used to
generate answers — separate from the Vertex judge model used to score them).
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from groq import RateLimitError
from vertexai.evaluation import EvalTask

from src.config.config import Config
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

EXPERIMENT_NAME = "rag-answer-quality-eval"

# ---------------------------------------------------------------------------
# Fixed question set. Deliberately the same 25 questions as
# ci/retrieval_eval.py's TEST_CASES (kept as plain strings here — no keyword
# lists needed, since judging is done by the Gen AI eval service, not
# substring matching). If you add a question there, consider adding it here
# too so retrieval and answer-quality checks cover the same ground.
# ---------------------------------------------------------------------------
# Each entry is (question, reference_answer). The reference (golden) answers
# were sourced DIRECTLY from the NVIDIA FY2025 Annual Report PDF (fiscal year
# ended January 26, 2025), NOT from the app's own output — grading the model
# against its own answers would be circular. Page citations for the figures:
# income statement p129/p143, segment revenue p171, reportable segments p132,
# R&D/SG&A p133, tax/cash p134/p143, employees p102, geographic p170,
# buybacks p135. Several figures corrected here vs. the older retrieval-eval
# keyword lists (e.g. employees ~36,000 not 29,600; gaming $11,350M not 11,446;
# professional visualization $1,878M not 1,591) — the PDF is the source of truth.
QA_PAIRS = [
    ("What was NVIDIA's total revenue in fiscal year 2025?",
     "NVIDIA's total revenue for fiscal year 2025 was $130.5 billion ($130,497 million), up 114% from $60.9 billion in fiscal 2024."),
    ("What was NVIDIA's net income in FY2025?",
     "NVIDIA's net income for fiscal year 2025 was $72,880 million (about $72.9 billion), up 145% from $29,760 million in fiscal 2024."),
    ("What were NVIDIA's earnings per share in FY2025?",
     "NVIDIA's net income per diluted share for fiscal year 2025 was $2.94, up 147% from $1.19 in fiscal 2024."),
    ("What were NVIDIA's sales, general and administrative expenses in FY2025?",
     "NVIDIA's sales, general and administrative (SG&A) expenses for fiscal year 2025 were $3,491 million, up 32% from $2,654 million in fiscal 2024."),
    ("What was NVIDIA's income tax expense in FY2025?",
     "NVIDIA's income tax expense for fiscal year 2025 was $11,146 million (about $11.1 billion), on income before tax of $84,026 million, an effective tax rate of roughly 13.3%."),
    ("How much cash did NVIDIA generate from operating activities in FY2025?",
     "NVIDIA generated $64,089 million (about $64.1 billion) in net cash provided by operating activities in fiscal year 2025, up from $28,090 million in fiscal 2024."),
    ("What were NVIDIA's total cash, cash equivalents, and marketable securities at end of FY2025?",
     "At the end of fiscal year 2025 (January 26, 2025), NVIDIA had $43,210 million (about $43.2 billion) in cash, cash equivalents, and marketable securities — $8,589 million in cash and cash equivalents plus $34,621 million in marketable securities."),
    ("What was NVIDIA's data center segment revenue in FY2025?",
     "NVIDIA's Data Center revenue for fiscal year 2025 was $115,186 million (about $115.2 billion), up 142% from a year earlier."),
    ("What was NVIDIA's gaming revenue in FY2025?",
     "NVIDIA's Gaming revenue for fiscal year 2025 was $11,350 million (about $11.4 billion), up 9% from a year earlier."),
    ("What was NVIDIA's professional visualization revenue?",
     "NVIDIA's Professional Visualization revenue for fiscal year 2025 was $1,878 million, up 21% from a year earlier."),
    ("What was NVIDIA's automotive segment revenue in FY2025?",
     "NVIDIA's Automotive revenue for fiscal year 2025 was $1,694 million, up 55% from a year earlier, driven by sales of self-driving platforms."),
    ("What was NVIDIA's Compute and Networking segment operating income in FY2025?",
     "NVIDIA's Compute & Networking reportable segment had operating income of $82,875 million in fiscal year 2025, up 159% from $32,016 million in fiscal 2024."),
    ("What is the Blackwell GPU architecture?",
     "Blackwell is NVIDIA's GPU architecture that succeeds Hopper, designed for accelerated computing and generative AI. Blackwell-based products include the B100, B200, and the GB200 Grace Blackwell superchip, delivering large gains in AI training and inference performance."),
    ("What products use the Hopper architecture?",
     "NVIDIA's Hopper architecture powers its H100 and H200 data center GPUs, which are used for AI training and inference and large-language-model workloads."),
    ("What is NVLink and how does it work?",
     "NVLink is NVIDIA's high-speed GPU-to-GPU interconnect that lets multiple GPUs communicate with far higher bandwidth than PCIe. Combined with NVSwitch, it connects many GPUs into a single high-bandwidth compute fabric for large-scale AI and HPC workloads."),
    ("What is CUDA and why is it important to NVIDIA?",
     "CUDA is NVIDIA's parallel computing platform and programming model that lets developers use NVIDIA GPUs for general-purpose computing. It is central to NVIDIA's strategy because its large software ecosystem and developer base create a durable moat around NVIDIA's accelerated-computing platform."),
    ("What automotive products does NVIDIA offer?",
     "NVIDIA's automotive products include the DRIVE platform for autonomous vehicles, the DRIVE Orin system-on-chip, and Jetson edge modules, supporting self-driving and in-vehicle AI."),
    ("What is NVIDIA's strategy for accelerated computing?",
     "NVIDIA's strategy is to provide a full-stack accelerated-computing platform — GPUs, networking, CUDA and other software — spanning the data center, so customers can run AI and other demanding workloads far more efficiently than on general-purpose CPUs."),
    ("What are the main risks NVIDIA faces from competition?",
     "NVIDIA cites competition from companies such as AMD and Intel, from cloud providers and other customers developing their own in-house chips, and the risk that rivals' products, pricing, or ecosystems erode its market position in GPUs and accelerated computing."),
    ("What export controls affect NVIDIA's China business?",
     "U.S. government export controls restrict sales of certain advanced NVIDIA data center GPUs to China and other regions, requiring licenses for some products. These controls have reduced NVIDIA's China Data Center revenue as a percentage of total to below pre-October-2023 levels and create ongoing uncertainty."),
    ("Who manufactures NVIDIA chips?",
     "NVIDIA is fabless and relies on third-party foundries to manufacture its chips, principally Taiwan Semiconductor Manufacturing Company (TSMC), along with other suppliers for assembly, packaging, and testing."),
    ("How many employees does NVIDIA have?",
     "As of the end of fiscal year 2025, NVIDIA had approximately 36,000 employees across 38 countries."),
    ("How much did NVIDIA return to shareholders in FY2025?",
     "In fiscal year 2025 NVIDIA returned capital to shareholders primarily through share repurchases, buying back 310 million shares for $34.0 billion, in addition to cash dividends."),
    ("What is NVIDIA's R&D spending?",
     "NVIDIA's research and development expense for fiscal year 2025 was $12,914 million (about $12.9 billion), up 49% from $8,675 million in fiscal 2024."),
    ("What percentage of NVIDIA revenue comes from outside the United States?",
     "Roughly 53% of NVIDIA's fiscal year 2025 revenue came from outside the United States (U.S. revenue was $61,257 million of $130,497 million total, about 47%), with significant revenue billed to Singapore, Taiwan, and China."),
]

QUESTIONS = [q for q, _ in QA_PAIRS]
REFERENCE_ANSWERS = [ref for _, ref in QA_PAIRS]

# Deterministic key-figure correctness targets, aligned index-for-index with
# QA_PAIRS. An answer is "correct" if it contains ANY accepted form for its
# question (case-insensitive substring). All figures are PDF-verified (see the
# QA_PAIRS provenance note). This replaces Vertex's removed
# question_answering_correctness with an objective, API-free, 0-1-scale check —
# analogous to unit-tests-as-eval for answers that have exact expected values.
# The financial 10 (indices 0-9) are the ones the CI gate actually runs under
# --limit=10; the rest use key terms and are here for completeness / local runs.
    # For each figure, accept the millions form ("130,497"), the rounded-billions
    # form ("130.5"), AND the exact-billions form the model sometimes emits
    # ("130.497" / "72.88"). Learned from the first run: qwen answered net income
    # as "$72.88 billion", which is exactly $72,880M but didn't match "72.9".
KEY_FIGURES = [
    ["130,497", "130.5", "130.4"],           # 0 total revenue ($130.497B)
    ["72,880", "72.9", "72.88"],             # 1 net income ($72.88B)
    ["2.94"],                                # 2 diluted EPS
    ["3,491"],                               # 3 SG&A
    ["11,146", "11.1", "11.15"],             # 4 income tax expense
    ["64,089", "64.1", "64.09"],             # 5 operating cash flow
    ["43,210", "43.2", "43.21"],             # 6 cash + marketable securities
    ["115,186", "115.2", "115.19"],          # 7 data center revenue
    ["11,350", "11.4", "11.35"],             # 8 gaming revenue
    ["1,878", "1.9", "1.88"],                # 9 professional visualization revenue
    ["1,694", "1.7", "1.69"],                # 10 automotive revenue
    ["82,875", "82.9", "82.88"],             # 11 compute & networking op income
    ["blackwell"],                           # 12
    ["hopper", "h100", "h200"],              # 13
    ["nvlink"],                              # 14
    ["cuda"],                                # 15
    ["drive", "orin", "jetson"],             # 16
    ["accelerated computing"],               # 17
    ["amd", "intel", "competition"],         # 18
    ["export", "china", "license"],          # 19
    ["tsmc", "taiwan semiconductor"],        # 20
    ["36,000"],                              # 21 employees
    ["34.0", "34,000", "310 million"],       # 22 buybacks
    ["12,914", "12.9"],                      # 23 R&D
    ["53%", "47%", "outside the united states", "international"],  # 24 geographic
]


def key_figure_correctness(responses, limit: int = 0):
    """Fraction of responses containing an accepted key figure for their question.

    Deterministic, no API call, 0-1 scale (mean of per-answer 0/1). `responses`
    is aligned to QUESTIONS[:limit]. Returns (mean, per_item_detail).
    """
    n = len(responses)
    targets = KEY_FIGURES[:n]
    detail = []
    hits = 0
    for i, resp in enumerate(responses):
        text = (resp or "").lower()
        accepted = [t.lower() for t in targets[i]]
        ok = any(a in text for a in accepted)
        hits += int(ok)
        detail.append((i, ok, targets[i]))
    mean = hits / n if n else 0.0
    return mean, detail


# Groq free/on-demand tier caps qwen3.6-27b at 8000 TPM (tokens per minute).
# The eval fires the full multi-call graph per question, so back-to-back runs
# blow the rolling window (a real run died at question ~12 with a 429). Two
# guards keep it under the cap with minimal added time:
#   1. A small fixed sleep BETWEEN questions to smooth the token rate.
#   2. A 429-aware retry that waits exactly the delay Groq asks for (parsed from
#      the error message, e.g. "try again in 8.25s") rather than a blanket sleep.
# The retry only costs time on the occasional question that still clips the
# window, so the common-case overhead is just (N-1) * INTER_QUESTION_DELAY.
INTER_QUESTION_DELAY = 4.0   # seconds between questions
MAX_RATE_LIMIT_RETRIES = 5


def _extract_retry_after(err: RateLimitError, default: float = 10.0) -> float:
    """Pull the 'try again in Xs' hint from a Groq 429, else fall back."""
    msg = str(getattr(err, "message", "") or err)
    m = re.search(r"try again in ([\d.]+)\s*s", msg, re.IGNORECASE)
    if m:
        # Add a small cushion so we're safely past the window edge.
        return float(m.group(1)) + 1.0
    return default


def _run_with_rate_limit_retry(graph, question: str) -> dict:
    """graph.run(question), retrying on Groq 429s with the server-suggested wait."""
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return graph.run(question)
        except RateLimitError as e:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            wait = _extract_retry_after(e)
            print(f"    rate limited (attempt {attempt}/{MAX_RATE_LIMIT_RETRIES}); "
                  f"waiting {wait:.1f}s then retrying...")
            time.sleep(wait)


def _answer_via_api(api_url: str, api_key: str, question: str):
    """POST /query to the deployed backend; return (answer, context_string).

    This is the eval-fidelity mode (item 8): grade the EXACT deployed endpoint
    users hit, not a pipeline rebuilt in-process. The API returns `sources`
    (list of {text, source, page}) instead of Document objects.
    """
    import requests
    headers = {"X-API-Key": api_key} if api_key else {}
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        resp = requests.post(
            f"{api_url.rstrip('/')}/query",
            json={"question": question, "history": []},
            headers=headers, timeout=120,
        )
        if resp.status_code == 429:  # backend rate-limited; back off
            if attempt == MAX_RATE_LIMIT_RETRIES:
                resp.raise_for_status()
            time.sleep(10.0)
            continue
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        context = "\n\n".join(s.get("text", "") for s in (data.get("sources") or [])[:8])
        return answer, context
    return "", ""


def build_eval_dataset(limit: int = 0, api_url: str = "", api_key: str = "") -> pd.DataFrame:
    """
    Build the {prompt, response, reference} DataFrame the Gen AI eval service
    needs, by answering each question either IN-PROCESS (default) or against a
    deployed /query endpoint (api_url set — item 8 eval-fidelity mode).

    prompt = question + retrieved context (the evaluator needs to see what
    information the model had access to). response = the final, post-guardrail
    answer, exactly what a user would see.

    `limit` > 0 evaluates only the first N question/reference pairs (kept in
    lockstep) — used to fit under Groq's TPM cap without a full 25-question run.
    """
    questions = QUESTIONS[:limit] if limit and limit > 0 else QUESTIONS
    references = REFERENCE_ANSWERS[:limit] if limit and limit > 0 else REFERENCE_ANSWERS

    graph = None
    if not api_url:
        # In-process mode: build the pipeline locally. Retrieval is served by
        # Qdrant Cloud (item 9) — no local index to load.
        llm = Config.get_llm()
        vs = VectorStore()
        retriever = vs.get_hybrid_retriever(k=8, rerank_top_k=5)
        graph = GraphBuilder(retriever, llm)

    prompts = []
    responses = []

    for i, question in enumerate(questions):
        if api_url:
            answer, context = _answer_via_api(api_url, api_key, question)
        else:
            result = _run_with_rate_limit_retry(graph, question)
            answer = result.get("answer", "")
            retrieved_docs = result.get("retrieved_docs", [])
            context = "\n\n".join(d.page_content for d in retrieved_docs[:8])

        prompt = f"Answer the question: {question}\n\nContext:\n{context}"
        prompts.append(prompt)
        responses.append(answer)

        print(f"  [{len(prompts)}/{len(questions)}] '{question[:60]}' -> "
              f"'{answer[:80]}'")

        # Space out questions to stay under the TPM cap (skip after the last).
        if i < len(questions) - 1:
            time.sleep(INTER_QUESTION_DELAY)

    # `reference` column = the golden answers, used by question_answering_correctness
    # (reference-based). groundedness / qa_quality ignore it (reference-free).
    return pd.DataFrame({
        "prompt": prompts,
        "response": responses,
        "reference": references,
    })


def run_eval(fail_under_groundedness: float, fail_under_qa_quality: float,
             fail_under_qa_correctness: float, limit: int = 0,
             api_url: str = "", api_key: str = "") -> int:
    n = limit if (limit and limit > 0) else len(QUESTIONS)
    print("NVIDIA RAG — Answer Quality Evaluation (CI gate)")
    print("=" * 60)
    print(f"Question set: {n}" + (f" (limited from {len(QUESTIONS)})" if n != len(QUESTIONS) else ""))
    if api_url:
        print(f"Mode: DEPLOYED API ({api_url}) — grading the live /query endpoint")
    else:
        print("Mode: in-process pipeline (rewriter -> responder -> guardrail)")
    print()

    dataset = build_eval_dataset(limit=limit, api_url=api_url, api_key=api_key)

    print()
    print("Dataset built. Scoring with Vertex AI Gen AI evaluation service...")

    # question_answering_correctness is reference-based (uses the `reference`
    # column of golden answers). groundedness + qa_quality are reference-free.
    # NOTE: Vertex's built-in question_answering_correctness was REMOVED from the
    # Gen AI eval service (confirmed 2026-07-15: "Metric name: ... is not
    # supported"). Reference-based correctness is now computed locally by
    # key_figure_correctness() below — deterministic, no API call, 0-1 scale.
    eval_task = EvalTask(
        dataset=dataset,
        metrics=[
            "groundedness",
            "question_answering_quality",
        ],
        experiment=EXPERIMENT_NAME,
    )
    result = eval_task.evaluate()

    summary = result.summary_metrics
    # Vertex's summary_metrics keys look like "<metric>/mean" — pull those.
    groundedness_mean = summary.get("groundedness/mean", 0.0)
    qa_quality_mean = summary.get("question_answering_quality/mean", 0.0)

    # Deterministic reference-based correctness on the same rows: fraction of
    # answers containing an accepted key figure for their question. Uses the
    # dataset's `response` column (the final, post-guardrail answers).
    qa_correctness_mean, correctness_detail = key_figure_correctness(
        dataset["response"].tolist(), limit=limit
    )

    print()
    print("=" * 60)
    print(f"Groundedness (mean)              : {groundedness_mean:.2f}  "
          f"(gate >= {fail_under_groundedness:.2f})")
    print(f"Question answering quality (mean): {qa_quality_mean:.2f}  "
          f"(gate >= {fail_under_qa_quality:.2f})")
    # Key-figure correctness: 0-1 scale (fraction of answers containing an
    # accepted PDF-verified figure/term for their question). Deterministic, no
    # API call — replaces Vertex's removed question_answering_correctness.
    print(f"Key-figure correctness (0-1)     : {qa_correctness_mean:.3f}  "
          f"(gate >= {fail_under_qa_correctness:.2f}"
          f"{' — DISABLED' if fail_under_qa_correctness <= 0 else ''})")
    misses = [(i, tgt) for (i, ok, tgt) in correctness_detail if not ok]
    if misses:
        print(f"  correctness misses ({len(misses)}):")
        for i, tgt in misses:
            print(f"    Q{i+1} '{QUESTIONS[i][:50]}' — expected one of {tgt}")
    print()

    failed = []
    if groundedness_mean < fail_under_groundedness:
        failed.append(f"groundedness {groundedness_mean:.2f} < "
                       f"{fail_under_groundedness:.2f}")
    if qa_quality_mean < fail_under_qa_quality:
        failed.append(f"question_answering_quality {qa_quality_mean:.2f} < "
                       f"{fail_under_qa_quality:.2f}")
    if fail_under_qa_correctness > 0 and qa_correctness_mean < fail_under_qa_correctness:
        failed.append(f"question_answering_correctness {qa_correctness_mean:.3f} < "
                       f"{fail_under_qa_correctness:.2f}")

    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1

    print("PASS: all gated metrics meet their gates")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-under-groundedness",
        type=float,
        default=0.7,
        help="Exit 1 if mean groundedness score falls below this. Scale: "
             "0-1 (binary per-example, confirmed via Vertex's predefined "
             "metric docs — groundedness is scored 0 or 1 per response, "
             "so the mean is a fraction grounded). Default: 0.7.",
    )
    parser.add_argument(
        "--fail-under-qa-quality",
        type=float,
        default=3.5,
        help="Exit 1 if mean question_answering_quality score falls below "
             "this. Scale: 1-5 (rating rubric, 5=best — confirmed via "
             "Vertex's metrics-templates docs; NOT 0-1, unlike "
             "groundedness). First real run scored 4.76/5. Default: 3.5.",
    )
    parser.add_argument(
        "--fail-under-qa-correctness",
        type=float,
        default=0.0,
        help="Exit 1 if key-figure correctness falls below this. Scale: 0-1 "
             "(fraction of answers containing an accepted PDF-verified figure/"
             "term for their question — deterministic, no API call; replaces "
             "Vertex's removed question_answering_correctness). Default 0.0 = "
             "gate reported but not enforced on the first run; the CI yaml sets "
             "an explicit value (e.g. 0.8) once the baseline is known.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N question/reference pairs (0 = all). "
             "Use for a quick subset run — e.g. to read a new metric's scale "
             "on a slice that fits under Groq's TPM cap without a full run.",
    )
    parser.add_argument(
        "--api-url",
        default="",
        help="If set, grade the DEPLOYED /query endpoint (item 8 eval-fidelity "
             "mode) instead of rebuilding the pipeline in-process. Needs no local "
             "FAISS index. E.g. https://rag-api-xxxx.run.app",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="X-API-Key header value for --api-url (the shared rag-api key). "
             "Can also be read from the API_KEY env var.",
    )
    args = parser.parse_args()
    import os as _os
    sys.exit(run_eval(
        args.fail_under_groundedness,
        args.fail_under_qa_quality,
        args.fail_under_qa_correctness,
        args.limit,
        args.api_url,
        args.api_key or _os.getenv("API_KEY", ""),
    ))
