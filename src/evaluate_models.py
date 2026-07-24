"""
Evaluate a base SentenceTransformer model against a fine-tuned checkpoint
using triplet-based cosine similarity margins.

Usage:
    python src/evaluate_models.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# =============================================================================
# CLI ARGUMENTS
# =============================================================================
# Point --finetuned-model at whichever checkpoint (or the final saved model)
# you want to evaluate, e.g.:
#   python src/evaluate_models.py --finetuned-model models/fine_tuned_all_mini/checkpoint-600
#   python src/evaluate_models.py --finetuned-model models/final_best_model_all_mini

parser = argparse.ArgumentParser(description="Evaluate base vs. fine-tuned SentenceTransformer models.")
parser.add_argument("--eval-data", default="data/eval_triplets.xlsx",
                     help="Path to evaluation triplets (.xlsx with ANCHOR/POSITIVE/NEGATIVE columns)")
parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Base (pre-fine-tuning) model name or path")
parser.add_argument("--finetuned-model", default="models/final_best_model_all_mini",
                     help="Path to the fine-tuned model or checkpoint to evaluate")
parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
parser.add_argument("--margins", type=float, nargs="+", default=[0.30, 0.20, 0.10],
                     help="Cosine-similarity margins to evaluate pass/fail accuracy at")
args = parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================

EVAL_DATA_PATH = args.eval_data

BASE_MODEL_NAME = args.base_model

FINETUNED_MODEL_PATH = args.finetuned_model

BATCH_SIZE = args.batch_size

MARGINS = args.margins


# =============================================================================
# LOAD EVALUATION DATA
# =============================================================================

evaluation = pd.read_excel(EVAL_DATA_PATH)
evaluation.drop(columns=["GROUP_ID"], inplace=True, errors="ignore")

print("Evaluation Columns:")
print(evaluation.columns)

anchors = evaluation["ANCHOR"].tolist()
positives = evaluation["POSITIVE"].tolist()
negatives = evaluation["NEGATIVE"].tolist()

print(f"\nEvaluation Dataset Size: {len(evaluation)}")


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model_path, model_name):
    """
    Evaluate a SentenceTransformer model on the evaluation dataset.
    """

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    model = SentenceTransformer(model_path)

    # -------------------------------------------------------------------------
    # Generate normalized embeddings
    # -------------------------------------------------------------------------

    anchor_embeddings = model.encode(
        anchors,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    positive_embeddings = model.encode(
        positives,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    negative_embeddings = model.encode(
        negatives,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # -------------------------------------------------------------------------
    # Compute cosine similarities
    # Since embeddings are normalized, dot product = cosine similarity
    # -------------------------------------------------------------------------

    positive_scores = torch.sum(anchor_embeddings * positive_embeddings, dim=1)
    negative_scores = torch.sum(anchor_embeddings * negative_embeddings, dim=1)

    score_gap = positive_scores - negative_scores

    results = pd.DataFrame({
        "POSITIVE_SCORE": positive_scores.cpu().numpy(),
        "NEGATIVE_SCORE": negative_scores.cpu().numpy(),
        "SCORE_GAP": score_gap.cpu().numpy(),
    })

    # -------------------------------------------------------------------------
    # Margin Evaluation
    # -------------------------------------------------------------------------

    summary = []

    for margin in MARGINS:
        passed = (results["SCORE_GAP"] > margin).sum()
        total = len(results)
        failed = total - passed
        accuracy = passed / total

        summary.append({
            "MARGIN": margin,
            "PASSED": int(passed),
            "FAILED": int(failed),
            "TOTAL": total,
            "ACCURACY": accuracy,
        })

    summary_df = pd.DataFrame(summary)

    print(f"\nTotal Evaluation Triplets: {len(results)}\n")

    for _, row in summary_df.iterrows():
        print(
            f"Margin {row['MARGIN']:.2f}: "
            f"{int(row['PASSED'])}/{int(row['TOTAL'])} passed, "
            f"{int(row['FAILED'])} failed "
            f"({row['ACCURACY']:.2%})"
        )

    print("\nMargin Summary:")
    print(summary_df)

    return summary_df


# =============================================================================
# BASE MODEL EVALUATION
# =============================================================================

base_summary = evaluate_model(BASE_MODEL_NAME, "BASE MODEL RESULTS")


# =============================================================================
# FINE-TUNED MODEL EVALUATION
# =============================================================================

finetuned_summary = evaluate_model(FINETUNED_MODEL_PATH, "FINE-TUNED MODEL RESULTS")
