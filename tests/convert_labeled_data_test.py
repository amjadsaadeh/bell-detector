import unittest
from pathlib import Path
import pandas as pd
from convert_labeled_data import annotation_to_sample_per_row


BASE_DATA_PATH = Path(__file__).parent / 'data'


class TestConvertLabeledData(unittest.TestCase):

    def test_annotation_to_sample_per_row(self):
        df = pd.read_csv(BASE_DATA_PATH / 'labeled_data.csv')
        gt_df = pd.read_csv(BASE_DATA_PATH / 'annotation_per_row_data.csv')  # Data to test against
        converted_df = annotation_to_sample_per_row(df)

        for series_name, series in converted_df.items():
            self.assertIn(series_name, gt_df.columns, f'Column {series_name} not found in reference DataFrame')
            pd.testing.assert_series_equal(series, gt_df[series_name], f'Column {series_name} is not equal')


if __name__ == '__main__':
    unittest.main()
    