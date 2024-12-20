"""
This script is for preparation of meta data received by label studio.
It consumes the CSV from label studio (each line is a file/task) and converts it into a CSV whith each sample as line (sample = labeled interval).
"""

import pandas as pd
import json


def annotation_to_sample_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """Coverts a DataFrame with soundfile per row into an annotation per row format.

    Args:
        df (pd.DataFrame): DataFrame with soundfile per row format

    Returns:
        pd.DataFrame: Dataframe with annotation per row format
    """

    result = {
        "annotation_id": list(),
        "audio_path": list(),
        "file_id": list(),
        "start": list(),
        "end": list(),
        "label": list(),
    }

    for row in df.iterrows():
        row = row[1]
        annotation_data = json.loads(row.label)

        for annotation in annotation_data:
            for label in annotation["labels"]:
                result["label"].append(label)
                result["start"].append(annotation["start"])
                result["end"].append(annotation["end"])
                result["file_id"].append(row.id)
                result["audio_path"].append(row.audio)
                result["annotation_id"].append(row.annotation_id)

    return pd.DataFrame(result)


if __name__ == "__main__":

    data_file_path = "./data/labeled_data.csv"
    target_file_path = "./data/annotation_per_row_data.csv"

    df = pd.read_csv(data_file_path)
    converted_df = annotation_to_sample_per_row(df)
    converted_df.to_csv(target_file_path)
