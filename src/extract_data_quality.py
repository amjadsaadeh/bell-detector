"""
This script is used to extract data quality metrics and saves them in a json file.
"""
import json
import pandas as pd


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/annotation_per_row_data.csv')

    samples_per_label = df.groupby('label').size()

    # Calculate imbalance between labels
    max_samples = samples_per_label.max()
    min_samples = samples_per_label.min()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else None

    # Calculate data quality metrics
    data_quality = {
        'total_samples': len(df),
        'total_files': len(df.file_id.unique()),
        'total_labels': len(df.label.unique()),
        'imbalance_ratio': imbalance_ratio
    }

    # Save data quality metrics
    with open('data/data_quality/general.json', 'w') as f:
        json.dump(data_quality, f, indent=4)
    
    samples_per_label.to_csv('data/data_quality/samples_per_label.csv')
