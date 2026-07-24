"""
Generate a small synthetic triplet dataset so the pipeline can be run
end-to-end immediately, without needing your own labeled data first.

Usage:
    python src/generate_sample_data.py
"""

import random
import pandas as pd

random.seed(42)

FIRST_NAMES = ["John", "Jon", "Jonathan", "Emma", "Emily", "Michael", "Mike",
               "Sarah", "Sara", "David", "Dave", "Priya", "Priyanka", "Wei",
               "Wei Wei", "Carlos", "Carl", "Anna", "Ana", "Robert", "Rob"]

COMPANIES = ["Acme Corp", "Acme Corporation", "Globex Ltd", "Globex Limited",
             "Initech Inc", "Initech", "Umbrella LLC", "Umbrella Co",
             "Stark Industries", "Stark Ind.", "Wayne Enterprises", "Wayne Ent"]

CITIES = ["New York", "NY", "Los Angeles", "LA", "San Francisco", "SF",
          "Chicago", "Boston", "Seattle", "Austin", "Miami", "Denver"]


def make_record(name, company, city):
    return f"{name} | {company} | {city}"


def generate_triplets(n):
    rows = []
    for _ in range(n):
        # Anchor and positive: same underlying entity, different formatting
        name = random.choice(FIRST_NAMES)
        company = random.choice(COMPANIES)
        city = random.choice(CITIES)

        anchor = make_record(name, company, city)
        positive = make_record(name, company, city)  # near-duplicate formatting

        # Negative: a genuinely different entity
        neg_name = random.choice([n for n in FIRST_NAMES if n != name])
        neg_company = random.choice([c for c in COMPANIES if c != company])
        neg_city = random.choice([c for c in CITIES if c != city])
        negative = make_record(neg_name, neg_company, neg_city)

        rows.append({"ANCHOR": anchor, "POSITIVE": positive, "NEGATIVE": negative})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    train_df = generate_triplets(300)
    eval_df = generate_triplets(60)

    train_df.to_excel("data/train_triplets.xlsx", index=False)
    eval_df.to_excel("data/eval_triplets.xlsx", index=False)

    print(f"Wrote {len(train_df)} training triplets to data/train_triplets.xlsx")
    print(f"Wrote {len(eval_df)} evaluation triplets to data/eval_triplets.xlsx")
    print("\nThis is toy/synthetic data meant only to verify the pipeline runs "
          "end-to-end. Replace it with your own labeled triplets for real use.")
