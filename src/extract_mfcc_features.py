import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
import dvc.api
import tqdm


AUDIO_DATA_PATH = Path('./data/audio')


def extract_mffc_features(file_path: Path | str, start: int, end: int, n_mfcc: int = 25, n_fft=2048) -> np.ndarray:
    # Extract MFCC features
    audio = AudioSegment.from_wav(AUDIO_DATA_PATH / file_path)
    info = mediainfo(AUDIO_DATA_PATH / file_path)
    audio_segment = np.array(audio[start:end].get_array_of_samples(), dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=int(info['sample_rate']), n_mfcc=n_mfcc, n_fft=n_fft)

    return mfccs

if __name__ == '__main__':
    params = dvc.api.params_show()
    chunked_data = pd.read_csv('./data/chunked_data.csv')
    
    # Create chunks using list comprehension instead of iterative append
    data_with_features = [
        row.to_dict() | {'mfcc_features': extract_mffc_features(row['audio_path'], row['start'], row['end'],
                                                                params['feature_extraction']['n_mfcc'],
                                                                params['feature_extraction']['n_fft'])}
        for _, row in tqdm.tqdm(chunked_data.iterrows(), total=chunked_data.shape[0])
    ]
    
    # Create DataFrame from list of dictionaries
    result = pd.DataFrame(data_with_features)
    result.to_parquet('./data/chunked_mffc_data.parquet', index=False)
