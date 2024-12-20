import json
import pandas as pd
from dataqualityutils import get_data_quality_metrics


if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv('./data/chunked_data.csv')

    # Split into background and non-background samples
    background_samples = df[df['label'] == 'background']
    non_background_samples = df[df['label'] != 'background']

    # Randomly sample from background class to match minority class size
    balanced_background = background_samples.sample(n=len(non_background_samples), random_state=42)

    # Combine balanced datasets
    balanced_df = pd.concat([balanced_background, non_background_samples])

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    chunks_per_label = balanced_df.groupby('label').size()
    chunks_per_label.to_csv('./data/data_quality/chunks_per_label.csv')

    chunk_data_quality = get_data_quality_metrics(balanced_df)
    with open('./data/data_quality/chunk_balanced_quality.json', 'w') as f:
        json.dump(chunk_data_quality, f, indent=4)

    # Save balanced dataset
    balanced_df.to_csv('./data/balanced_data.csv', index=False)
