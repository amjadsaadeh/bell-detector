import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from src.extract_mfcc_features import process_audio_data


BASE_DATA_PATH = Path(__file__).parent / 'data'


class TestExtractMFCCFeatures(unittest.TestCase):

    def setUp(self):
        self.params = {
            'feature_extraction': {
                'n_mfcc': 13,
                'n_fft': 512
            }
        }
        self.input_path = BASE_DATA_PATH / 'test_balanced_data.csv'
        self.output_path = BASE_DATA_PATH / 'test_mfcc_data.h5'

        # Create a sample input CSV file
        sample_data = {
            'audio_path': ['test_audio.wav', 'test_audio.wav'],
            'start': [0, 500],
            'end': [1000, 1500]
        }
        pd.DataFrame(sample_data).to_csv(self.input_path, index=False)

    def tearDown(self):
        # Clean up the test files
        if self.input_path.exists():
            self.input_path.unlink()
        if self.output_path.exists():
            self.output_path.unlink()

    def test_process_audio_data(self):
        process_audio_data(self.params, self.input_path, self.output_path, BASE_DATA_PATH)

        # Check if the output file is created
        self.assertTrue(self.output_path.exists(), "Output file was not created")

        # Load the output file and check its contents
        result_df = pd.read_hdf(self.output_path, key='data')
        self.assertIn('mfcc_features', result_df.columns, "MFCC features column not found in the output DataFrame")
        self.assertEqual(len(result_df), 2, "Output DataFrame does not have the expected number of rows")
        self.assertIsInstance(result_df['mfcc_features'].iloc[0], np.ndarray, "MFCC features are not stored as numpy arrays")


if __name__ == '__main__':
    unittest.main()
