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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
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
QUESTIONS = [
    "What was NVIDIA's total revenue in fiscal year 2025?",
    "What was NVIDIA's net income in FY2025?",
    "What were NVIDIA's earnings per share in FY2025?",
    "What were NVIDIA's sales, general and administrative expenses in FY2025?",
    "What was NVIDIA's income tax expense in FY2025?",
    "How much cash did NVIDIA generate from operating activities in FY2025?",
    "What were NVIDIA's total cash, cash equivalents, and marketable securities at end of FY2025?",
    "What was NVIDIA's data center segment revenue in FY2025?",
    "What was NVIDIA's gaming revenue in FY2025?",
    "What was NVIDIA's professional visualization revenue?",
    "What was NVIDIA's automotive segment revenue in FY2025?",
    "What was NVIDIA's Compute and Networking segment operating income in FY2025?",
    "What is the Blackwell GPU architecture?",
    "What products use the Hopper architecture?",
    "What is NVLink and how does it work?",
    "What is CUDA and why is it important to NVIDIA?",
    "What automotive products does NVIDIA offer?",
    "What is NVIDIA's strategy for accelerated computing?",
    "What are the main risks NVIDIA faces from competition?",
    "What export controls affect NVIDIA's China business?",
    "Who manufactures NVIDIA chips?",
    "How many employees does NVIDIA have?",
    "How much did NVIDIA return to shareholders in FY2025?",
    "What is NVIDIA's R&D spending?",
    "What percentage of NVIDIA revenue comes from outside the United States?",
]


def build_eval_dataset() -> pd.DataFrame:
    """
    Run the real rewriter -> responder -> guardrail graph for every question
    and build the {prompt, response} DataFrame the Gen AI eval service needs.

    prompt = question + retrieved context (per the documented pattern: the
    evaluator needs to see what information the model had access to, not
    just the bare question).
    response = the final, user-facing answer — i.e. post-guardrail, exactly
    what a real user would see, fallback text included if it fired.
    """
    llm = Config.get_llm()
    vs = VectorStore()
    vs.load(FAISS_INDEX_PATH)
    retriever = vs.get_hybrid_retriever(k=8, rerank_top_k=5)

    graph = GraphBuilder(retriever, llm)

    prompts = []
    responses = []

    for question in QUESTIONS:
        result = graph.run(question)
        answer = result.get("answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        context = "\n\n".join(d.page_content for d in retrieved_docs[:8])

        prompt = f"Answer the question: {question}\n\nContext:\n{context}"
        prompts.append(prompt)
        responses.append(answer)

        print(f"  [{len(prompts)}/{len(QUESTIONS)}] '{question[:60]}' -> "
              f"'{answer[:80]}'")

    return pd.DataFrame({"prompt": prompts, "response": responses})


def run_eval(fail_under_groundedness: float, fail_under_qa_quality: float) -> int:
    print("NVIDIA RAG — Answer Quality Evaluation (CI gate)")
    print("=" * 60)
    print(f"Question set: {len(QUESTIONS)}")
    print("Running live pipeline (rewriter -> responder -> guardrail) "
          "for each question...")
    print()

    dataset = build_eval_dataset()

    print()
    print("Dataset built. Scoring with Vertex AI Gen AI evaluation service...")

    eval_task = EvalTask(
        dataset=dataset,
        metrics=["groundedness", "question_answering_quality"],
        experiment=EXPERIMENT_NAME,
    )
    result = eval_task.evaluate()

    summary = result.summary_metrics
    # Vertex's summary_metrics keys look like "<metric>/mean" — pull those.
    groundedness_mean = summary.get("groundedness/mean", 0.0)
    qa_quality_mean = summary.get("question_answering_quality/mean", 0.0)

    print()
    print("=" * 60)
    print(f"Groundedness (mean)             : {groundedness_mean:.2f}  "
          f"(gate >= {fail_under_groundedness:.2f})")
    print(f"Question answering quality (mean): {qa_quality_mean:.2f}  "
          f"(gate >= {fail_under_qa_quality:.2f})")
    print()

    failed = []
    if groundedness_mean < fail_under_groundedness:
        failed.append(f"groundedness {groundedness_mean:.2f} < "
                       f"{fail_under_groundedness:.2f}")
    if qa_quality_mean < fail_under_qa_quality:
        failed.append(f"question_answering_quality {qa_quality_mean:.2f} < "
                       f"{fail_under_qa_quality:.2f}")

    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1

    print("PASS: both metrics meet their gates")
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
    args = parser.parse_args()
    sys.exit(run_eval(args.fail_under_groundedness, args.fail_under_qa_quality))
