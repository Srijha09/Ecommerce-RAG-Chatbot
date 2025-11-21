from typing import List, Set, Tuple
import math

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# ------------ Retrieval metrics ------------

def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(top_k)

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top_k = retrieved_ids[:k]

    def dcg(doc_ids: List[str]) -> float:
        score = 0.0
        for i, d in enumerate(doc_ids, start=1):
            rel = 1.0 if d in relevant_ids else 0.0
            if rel > 0:
                score += (2**rel - 1) / math.log2(i + 1)
        return score

    ideal_order = sorted(top_k, key=lambda d: d in relevant_ids, reverse=True)
    ideal = dcg(ideal_order)
    actual = dcg(top_k)
    if ideal == 0:
        return 0.0
    return actual / ideal


# ------------ Answer metrics ------------

_smooth = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def text_metrics(
    prediction: str,
    reference: str,
) -> Tuple[float, float, float]:
    """
    Returns (BLEU, ROUGE-L, METEOR) as floats in [0, 1].
    """

    # BLEU
    bleu = sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=_smooth,
    )

    # ROUGE-L (F1)
    rouge_scores = _rouge.score(reference, prediction)
    rouge_l_f = rouge_scores["rougeL"].fmeasure

    # METEOR
    meteor = meteor_score([reference], prediction)

    return float(bleu), float(rouge_l_f), float(meteor)