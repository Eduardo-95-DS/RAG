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

Deliberately NOT using question_answering_correctness (would require a
golden reference answer per question — a maintenance burden not taken on
for this first pass). Revisit if false-negative groundedness misses become
a problem in practice.

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

FAISS_INDEX_PATH = "faiss_index"
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


def build_eval_dataset(limit: int = 0) -> pd.DataFrame:
    """
    Run the real rewriter -> responder -> guardrail graph for every question
    and build the {prompt, response, reference} DataFrame the Gen AI eval
    service needs.

    prompt = question + retrieved context (per the documented pattern: the
    evaluator needs to see what information the model had access to, not
    just the bare question).
    response = the final, user-facing answer — i.e. post-guardrail, exactly
    what a real user would see, fallback text included if it fired.

    `limit` > 0 evaluates only the first N question/reference pairs (kept in
    lockstep). Used to read a new metric's scale on a small slice that fits
    comfortably under Groq's TPM cap, without a full 25-question run.
    """
    questions = QUESTIONS[:limit] if limit and limit > 0 else QUESTIONS
    references = REFERENCE_ANSWERS[:limit] if limit and limit > 0 else REFERENCE_ANSWERS

    llm = Config.get_llm()
    vs = VectorStore()
    vs.load(FAISS_INDEX_PATH)
    retriever = vs.get_hybrid_retriever(k=8, rerank_top_k=5)

    graph = GraphBuilder(retriever, llm)

    prompts = []
    responses = []

    for i, question in enumerate(questions):
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
             fail_under_qa_correctness: float, limit: int = 0) -> int:
    n = limit if (limit and limit > 0) else len(QUESTIONS)
    print("NVIDIA RAG — Answer Quality Evaluation (CI gate)")
    print("=" * 60)
    print(f"Question set: {n}" + (f" (limited from {len(QUESTIONS)})" if n != len(QUESTIONS) else ""))
    print("Running live pipeline (rewriter -> responder -> guardrail) "
          "for each question...")
    print()

    dataset = build_eval_dataset(limit=limit)

    print()
    print("Dataset built. Scoring with Vertex AI Gen AI evaluation service...")

    # question_answering_correctness is reference-based (uses the `reference`
    # column of golden answers). groundedness + qa_quality are reference-free.
    eval_task = EvalTask(
        dataset=dataset,
        metrics=[
            "groundedness",
            "question_answering_quality",
            "question_answering_correctness",
        ],
        experiment=EXPERIMENT_NAME,
    )
    result = eval_task.evaluate()

    summary = result.summary_metrics
    # Vertex's summary_metrics keys look like "<metric>/mean" — pull those.
    groundedness_mean = summary.get("groundedness/mean", 0.0)
    qa_quality_mean = summary.get("question_answering_quality/mean", 0.0)
    qa_correctness_mean = summary.get("question_answering_correctness/mean", 0.0)

    print()
    print("=" * 60)
    print(f"Groundedness (mean)              : {groundedness_mean:.2f}  "
          f"(gate >= {fail_under_groundedness:.2f})")
    print(f"Question answering quality (mean): {qa_quality_mean:.2f}  "
          f"(gate >= {fail_under_qa_quality:.2f})")
    # SCALE UNVERIFIED for correctness — printed with more precision so the first
    # run reveals whether it's 0-1 or 1-5. The gate is DISABLED by default
    # (--fail-under-qa-correctness=0) until the scale is confirmed; see the flag
    # help and known_issues.md's scale-verification rule.
    print(f"Question answering correctness   : {qa_correctness_mean:.3f}  "
          f"(gate >= {fail_under_qa_correctness:.2f}"
          f"{' — DISABLED' if fail_under_qa_correctness <= 0 else ''})")
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
        help="Exit 1 if mean question_answering_correctness (reference-based, "
             "uses the golden answers) falls below this. SCALE UNVERIFIED — this "
             "metric is no longer in Vertex's current metrics-templates doc, so "
             "the first run must reveal whether it's 0-1 or 1-5 before a gate is "
             "meaningful. DEFAULT 0.0 = gate DISABLED (measure-only); set to a "
             "matching-scale value once the scale is confirmed. Per the project's "
             "scale-verification rule, a passing number is not proof the gate is "
             "right until the scale is checked.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N question/reference pairs (0 = all). "
             "Use for a quick subset run — e.g. to read a new metric's scale "
             "on a slice that fits under Groq's TPM cap without a full run.",
    )
    args = parser.parse_args()
    sys.exit(run_eval(
        args.fail_under_groundedness,
        args.fail_under_qa_quality,
        args.fail_under_qa_correctness,
        args.limit,
    ))
