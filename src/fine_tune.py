"""
Fine-tune a SentenceTransformer model (all-MiniLM-L6-v2) using Triplet Loss
for a domain-specific text embedding / information retrieval task
(e.g. customer record deduplication, entity resolution).

Usage:
    python src/fine_tune.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse

import pandas as pd

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from sentence_transformers.evaluation import TripletEvaluator


# =============================================================================
# CLI ARGUMENTS
# =============================================================================
# All paths can be overridden from the command line, e.g.:
#   python src/fine_tune.py --train-data data/my_train.xlsx --eval-data data/my_eval.xlsx
# Defaults assume you've run `python src/generate_sample_data.py` first,
# or placed your own train/eval files at the paths below.

parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer with triplet loss.")
parser.add_argument("--train-data", default="data/train_triplets.xlsx",
                     help="Path to training triplets (.xlsx with ANCHOR/POSITIVE/NEGATIVE columns)")
parser.add_argument("--eval-data", default="data/eval_triplets.xlsx",
                     help="Path to evaluation triplets (.xlsx with ANCHOR/POSITIVE/NEGATIVE columns)")
parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Base model to fine-tune (any SentenceTransformer-compatible model)")
parser.add_argument("--output-dir", default="models/fine_tuned_all_mini",
                     help="Directory where training checkpoints are saved")
parser.add_argument("--final-model-path", default="models/final_best_model_all_mini",
                     help="Directory where the best fine-tuned model is saved")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Per-device train batch size")
parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
args = parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_DATA_PATH = args.train_data
EVAL_DATA_PATH = args.eval_data

MODEL_NAME = args.model_name

OUTPUT_DIR = args.output_dir
FINAL_MODEL_PATH = args.final_model_path


# =============================================================================
# LOAD TRAINING DATA
# =============================================================================

train_df = pd.read_excel(TRAIN_DATA_PATH)
train_df.drop(columns=["GROUP_ID"], inplace=True, errors="ignore")

print("Training Columns:")
print(train_df.columns)


# =============================================================================
# LOAD EVALUATION DATA
# =============================================================================

evaluation_df = pd.read_excel(EVAL_DATA_PATH)
evaluation_df.drop(columns=["GROUP_ID"], inplace=True, errors="ignore")

print("Evaluation Columns:")
print(evaluation_df.columns)


# =============================================================================
# CONVERT TO HUGGING FACE DATASETS
# =============================================================================

train_dataset = Dataset.from_pandas(
    train_df[["ANCHOR", "POSITIVE", "NEGATIVE"]],
    preserve_index=False,
)

eval_dataset = Dataset.from_pandas(
    evaluation_df[["ANCHOR", "POSITIVE", "NEGATIVE"]],
    preserve_index=False,
)

print(f"Training Data Size   : {train_dataset.shape}")
print(f"Evaluation Data Size : {eval_dataset.shape}")


# =============================================================================
# LOAD BASE MODEL
# =============================================================================

model = SentenceTransformer(MODEL_NAME)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

loss = losses.TripletLoss(
    model=model,
    distance_metric=losses.TripletDistanceMetric.COSINE,
    triplet_margin=0.3,
)


# =============================================================================
# EVALUATOR
# =============================================================================

evaluator = TripletEvaluator(
    anchors=eval_dataset["ANCHOR"],
    positives=eval_dataset["POSITIVE"],
    negatives=eval_dataset["NEGATIVE"],
    name="domain_dedup_model_01",
)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=args.learning_rate,

    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=8,

    num_train_epochs=args.epochs,
    warmup_ratio=0.1,

    eval_strategy="steps",
    eval_steps=200,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=None,

    load_best_model_at_end=True,
    metric_for_best_model="domain_dedup_model_01_cosine_accuracy",
    greater_is_better=True,

    logging_steps=10,

    seed=42,
    data_seed=42,

    report_to="none",
)


# =============================================================================
# TRAINER
# =============================================================================

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


# =============================================================================
# TRAIN MODEL
# =============================================================================

trainer.train()

print("\nBest Checkpoint:")
print(trainer.state.best_model_checkpoint)

print("\nBest Metric:")
print(trainer.state.best_metric)


# =============================================================================
# SAVE BEST MODEL
# =============================================================================

trainer.model.save_pretrained(FINAL_MODEL_PATH)

print("\nFinal model saved to:")
print(FINAL_MODEL_PATH)


# =============================================================================
# VERIFY SAVED MODEL
# =============================================================================

verification_model = SentenceTransformer(FINAL_MODEL_PATH)

print("\nModel verification successful.")
