"""
This script is used to extract data quality metrics and saves them in a json file.
"""

import json
import pandas as pd
from dataqualityutils import get_data_quality_metrics


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("./data/annotation_per_row_data.csv")

    samples_per_label = df.groupby("label").size()

    # Calculate data quality metrics
    data_quality = get_data_quality_metrics(df)

    # Save data quality metrics
    with open("./data/data_quality/sample_based_quality.json", "w") as f:
        json.dump(data_quality, f, indent=4)

    samples_per_label.to_csv("./data/data_quality/samples_per_label.csv")
