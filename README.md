# Domain-Specific Text Embedding Models for Information Retrieval

Fine-tuning a general-purpose sentence embedding model (`all-MiniLM-L6-v2`)
on domain-specific triplet data (anchor / positive / negative) to improve
retrieval and matching quality for a specific task — for example, customer
record deduplication or entity resolution — where off-the-shelf embeddings
underperform.

The project trains with **Triplet Loss** so that, after fine-tuning,
semantically matching records sit closer together in embedding space than
non-matching ones, and evaluates the result against the base model using a
margin-based accuracy metric.

## Why domain-specific fine-tuning?

General-purpose sentence embedding models are trained on broad web/text
corpora and don't always capture the specific notion of "similarity" a
downstream task needs (e.g. two customer records referring to the same
real-world entity despite differing formatting, abbreviations, or typos).
Fine-tuning on labeled triplets from your own domain teaches the model
that specific notion of similarity directly.

## Project structure

```
domain-specific-text-embedding-models/
├── src/
│   ├── generate_sample_data.py  # Creates a small synthetic dataset to test the pipeline
│   ├── fine_tune.py             # Fine-tunes the base model on triplet data
│   └── evaluate_models.py       # Compares base vs. fine-tuned model
├── data/                          # Train/eval triplet .xlsx files live here
├── models/                        # Fine-tuned checkpoints and final model saved here
├── requirements.txt
└── README.md
```

## Data format

Both `fine_tune.py` and `evaluate_models.py` expect Excel files with the
following columns:

| ANCHOR | POSITIVE | NEGATIVE |
|---|---|---|
| text A | text A's true match | text A's non-match |

- **ANCHOR** — the reference text
- **POSITIVE** — a text that should be considered a match / similar to the anchor
- **NEGATIVE** — a text that should be considered a non-match / dissimilar to the anchor

A `GROUP_ID` column, if present, is dropped automatically before training.

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd domain-specific-text-embedding-models

python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Quickstart (no data required)

To verify everything works end-to-end before plugging in real data, generate
a small synthetic dataset and run the full pipeline on it:

```bash
python src/generate_sample_data.py
python src/fine_tune.py
python src/evaluate_models.py --finetuned-model models/final_best_model_all_mini
```

This trains on 300 synthetic triplets — enough to confirm the pipeline runs,
not enough to produce a meaningful model. Replace the sample data with your
own for real results (see below).

## Using your own data

1. Place your training and evaluation triplet files in `data/`, named
   whatever you like, e.g.:
   ```
   data/train_triplets.xlsx
   data/eval_triplets.xlsx
   ```
   (These are the default filenames the scripts look for — see CLI options
   below to point at different files/locations.)

2. Fine-tune the model:
   ```bash
   python src/fine_tune.py
   ```
   This trains `all-MiniLM-L6-v2` with cosine-distance Triplet Loss for
   3 epochs, evaluating every 200 steps, and automatically keeps the
   best checkpoint (by triplet cosine accuracy) via
   `load_best_model_at_end=True`. The best model is saved to
   `models/final_best_model_all_mini`, and individual checkpoints are
   saved under `models/fine_tuned_all_mini/checkpoint-<step>`.

   Useful flags (all optional, shown with their defaults):
   ```bash
   python src/fine_tune.py \
     --train-data data/train_triplets.xlsx \
     --eval-data data/eval_triplets.xlsx \
     --model-name sentence-transformers/all-MiniLM-L6-v2 \
     --output-dir models/fine_tuned_all_mini \
     --final-model-path models/final_best_model_all_mini \
     --epochs 3 \
     --batch-size 16 \
     --learning-rate 2e-5
   ```

3. Evaluate base vs. fine-tuned model:
   ```bash
   python src/evaluate_models.py --finetuned-model models/final_best_model_all_mini
   ```
   This computes cosine similarity between anchor–positive and
   anchor–negative pairs for both models, then reports how many triplets
   pass at similarity margins of 0.30 / 0.20 / 0.10 — higher pass rates
   indicate the model separates matches from non-matches more confidently.

   To evaluate a specific checkpoint instead of the final model:
   ```bash
   python src/evaluate_models.py --finetuned-model models/fine_tuned_all_mini/checkpoint-600
   ```

   Useful flags (all optional, shown with their defaults):
   ```bash
   python src/evaluate_models.py \
     --eval-data data/eval_triplets.xlsx \
     --base-model sentence-transformers/all-MiniLM-L6-v2 \
     --finetuned-model models/final_best_model_all_mini \
     --batch-size 16 \
     --margins 0.30 0.20 0.10
   ```

No path in either script needs to be edited by hand — everything is
configurable via command-line flags, with sensible defaults matching the
folder structure above.

## Method summary

- **Base model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Loss:** Triplet Loss, cosine distance metric, margin = 0.3
- **Training:** 3 epochs, batch size 16, learning rate 2e-5, 10% warmup
- **Model selection:** best checkpoint by triplet cosine accuracy on the
  held-out evaluation set
- **Evaluation:** margin-based pass/fail accuracy at three thresholds
  (0.30, 0.20, 0.10), comparing base vs. fine-tuned embeddings

## Notes

- Training runs on CPU by default on machines without a CUDA GPU; on
  Apple Silicon Macs, PyTorch's MPS backend can be used for acceleration
  if configured.
- Model checkpoints are excluded from version control via `.gitignore`
  (they're typically large); commit code and configuration only, and
  distribute trained weights separately if needed (e.g. Hugging Face Hub,
  cloud storage).
