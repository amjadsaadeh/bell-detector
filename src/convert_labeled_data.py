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
        'annotation_ids': list(),
        'audio_paths': list(),
        'file_ids': list(),
        'starts': list(),
        'ends': list(),
        'labels': list()
    }

    for row in df.iterrows():
        annotation_data = json.loads(row[1].label)

        for annotation in annotation_data:
            for label in annotation['labels']:
                result['labels'].append(label)
                result['starts'].append(annotation['start'])
                result['ends'].append(annotation['end'])
                result['file_ids'].append(row.id)
                result['audio_paths'].append(row.audio)
                result['annotation_ids'].append(row.annotation_id)
    
    return df.DataFrame(result)


if __name__ == '__main__':
    
    data_file_path = './data/labeled_data.csv'

    df = pd.read_csv(data_file_path)
    annotation_to_sample_per_row(df)
