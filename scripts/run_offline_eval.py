import json
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import List, Dict, Any
from rag.evaluator import InlineLLMJudge

# Optional metrics libs (install if missing)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    sentence_bleu = None
    meteor_score = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


EVAL_DATA_PATH = Path("everstorm_eval_dataset.jsonl")
OUT_PATH = Path("data/offline_eval_results.json")


# ---------------------------
# Helpers
# ---------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_bleu(reference: str, prediction: str) -> float:
    if sentence_bleu is None:
        return float("nan")
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    smoothie = SmoothingFunction().method1
    try:
        return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))
    except ZeroDivisionError:
        return 0.0


def compute_meteor(reference: str, prediction: str) -> float:
    if meteor_score is None:
        return float("nan")

    # Simple whitespace tokenization; replace with nltk.word_tokenize if you prefer
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    try:
        # meteor_score expects: List[List[str]], List[str]
        return float(meteor_score([ref_tokens], pred_tokens))
    except Exception:
        return 0.0


def compute_rouge_l(reference: str, prediction: str) -> float:
    if rouge_scorer is None:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


def maybe_warn_missing_libs():
    missing = []
    if sentence_bleu is None or meteor_score is None:
        missing.append("nltk (for BLEU, METEOR)")
    if rouge_scorer is None:
        missing.append("rouge-score (for ROUGE-L)")
    if missing:
        print("\n[WARN] Missing metric packages:")
        for m in missing:
            print(f"  - {m}")
        print("Install them with, for example:")
        print("  pip install nltk rouge-score\n")


# ---------------------------
# Offline evaluation
# ---------------------------

def main():
    if not EVAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Eval dataset not found at {EVAL_DATA_PATH}")

    maybe_warn_missing_libs()

    examples = load_jsonl(EVAL_DATA_PATH)
    print(f"Loaded {len(examples)} examples from {EVAL_DATA_PATH}")

    # InlineLLMJudge internally:
    #  - runs RAGPipeline.ask()
    #  - builds judge prompt with context + answer
    #  - returns answer, label, cycles, sources
    judge = InlineLLMJudge(max_cycles=3)

    bleu_scores = []
    meteor_scores = []
    rouge_l_scores = []
    judge_labels = []

    detailed_results = []

    for idx, ex in enumerate(examples, start=1):
        ex_id = ex.get("id", str(idx))
        question = ex["question"]
        reference_answer = ex["answer"]

        print(f"\n=== [{idx}/{len(examples)}] Example {ex_id} ===")
        print("Q:", question)
        print("Ref:", reference_answer)

        # Run RAG + LLM-as-judge
        result = judge.evaluate_answer(question)
        model_answer = result["answer"]
        label = result["label"]
        judge_labels.append(label)

        print("Answer:", model_answer)
        print("Judge label:", label)

        # Text overlap metrics (classic)
        bleu = compute_bleu(reference_answer, model_answer)
        meteor = compute_meteor(reference_answer, model_answer)
        rouge_l = compute_rouge_l(reference_answer, model_answer)

        print(f"BLEU: {bleu:.4f}, METEOR: {meteor:.4f}, ROUGE-L: {rouge_l:.4f}")

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        rouge_l_scores.append(rouge_l)

        detailed_results.append(
            {
                "id": ex_id,
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": model_answer,
                "judge_label": label,
                "bleu": bleu,
                "meteor": meteor,
                "rouge_l": rouge_l,
                "sources": result.get("sources", []),
                "judge_cycles": result.get("cycles", []),
            }
        )

    # -------- Summary --------
    def safe_mean(xs: List[float]) -> float:
        xs = [x for x in xs if not (x != x)]  # filter NaN
        return float(mean(xs)) if xs else float("nan")

    avg_bleu = safe_mean(bleu_scores)
    avg_meteor = safe_mean(meteor_scores)
    avg_rouge_l = safe_mean(rouge_l_scores)

    label_counts = Counter(judge_labels)
    total = len(judge_labels) or 1

    correct_rate = label_counts.get("CORRECT", 0) / total
    halluc_rate = label_counts.get("HALLUCINATION", 0) / total
    incomplete_rate = label_counts.get("INCOMPLETE", 0) / total

    print("\n===== OFFLINE EVAL SUMMARY =====")
    print(f"Total examples: {len(examples)}")
    print("\nJudge label distribution:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt} ({cnt/total:.1%})")

    print("\nText overlap metrics (vs reference_answer):")
    print(f"  BLEU (avg):     {avg_bleu:.4f}")
    print(f"  METEOR (avg):   {avg_meteor:.4f}")
    print(f"  ROUGE-L (avg):  {avg_rouge_l:.4f}")

    print("\nJudge-derived metrics:")
    print(f"  Correct rate:        {correct_rate:.1%}")
    print(f"  Hallucination rate:  {halluc_rate:.1%}")
    print(f"  Incomplete rate:     {incomplete_rate:.1%}")

    # Save detailed JSON
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "num_examples": len(examples),
                    "label_counts": dict(label_counts),
                    "avg_bleu": avg_bleu,
                    "avg_meteor": avg_meteor,
                    "avg_rouge_l": avg_rouge_l,
                    "correct_rate": correct_rate,
                    "hallucination_rate": halluc_rate,
                    "incomplete_rate": incomplete_rate,
                },
                "results": detailed_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDetailed results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
