"""
This script is to draw the data from the bigger dataset. It also takes care about data balancing.
"""

import json
import pandas as pd
import yaml
from dataqualityutils import get_data_quality_metrics


if __name__ == "__main__":

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # Read the dataset
    df = pd.read_csv("./data/annotation_per_row_data.csv")

    # Split into background and non-background samples
    background_samples = df[df["label"] == "background"]
    non_background_samples = df[df["label"] != "background"]

    # Cut in chunks
    chunk_size = params["chunk_size"]
    chunk_overlap = params["chunk_overlap"]

    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")

    # Create chunks using list comprehension instead of iterative append
    chunks = [
        row.to_dict() | {"start": chunk_start, "end": (chunk_start + chunk_size)}
        for _, row in annotated_data.iterrows()
        for chunk_start in range(
            int(row["start"] * 1000),  # convert to ms
            int(row["end"] * 1000) - chunk_size,
            chunk_overlap,
        )
    ]

    # Create DataFrame from list of dictionaries
    result = pd.DataFrame(chunks)
    result.to_csv("./data/chunked_data.csv", index=False)

    # Randomly sample from background class to match minority class size
    balanced_background = background_samples.sample(
        n=len(non_background_samples), random_state=42
    )

    # Combine balanced datasets
    balanced_df = pd.concat([balanced_background, non_background_samples])

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    chunks_per_label = balanced_df.groupby("label").size()
    chunks_per_label.to_csv("./data/data_quality/chunks_per_label.csv")

    chunk_data_quality = get_data_quality_metrics(balanced_df)
    with open("./data/data_quality/chunk_balanced_quality.json", "w") as f:
        json.dump(chunk_data_quality, f, indent=4)

    # Save balanced dataset
    balanced_df.to_csv("./data/balanced_data.h5", index=False)
