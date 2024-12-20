from typing import Dict
import pandas as pd


def get_data_quality_metrics(df: pd.DataFrame) -> Dict[str, int | float]:
    # Calculate imbalance between labels
    samples_per_label = df.groupby('label').size()

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

    return data_quality
