"""
This script is used to extract data quality metrics and saves them in a json file.
"""
import json
import pandas as pd


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/annotation_per_row_data.csv')

    samples_per_label = df.groupby('label').size().to_dict()

    # Calculate imbalance between labels
    max_samples = max(samples_per_label.values())
    min_samples = min(samples_per_label.values())
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else None

    # Calculate data quality metrics
    data_quality = {
        'total_samples': len(df),
        'total_files': len(df.file_id.unique()),
        'total_labels': len(df.label.unique()),
        'samples_per_label': samples_per_label,
        'imbalance_ratio': imbalance_ratio
    }

    # Save data quality metrics
    with open('data/data_quality.json', 'w') as f:
        json.dump(data_quality, f, indent=4)
