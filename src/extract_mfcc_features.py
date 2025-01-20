import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
import yaml
import tqdm
import functools
from multiprocessing import Pool


AUDIO_DATA_PATH = Path("./data/audio")


def extract_mffc_features(
    file_path: Path | str,
    start: int,
    end: int,
    n_mfcc: int = 25,
    n_fft=2048,
) -> np.ndarray:
    # Extract MFCC features
    audio = AudioSegment.from_wav(file_path)
    info = mediainfo(file_path)
    audio_segment = np.array(audio[start:end].get_array_of_samples(), dtype=np.float32)
    mfccs = librosa.feature.mfcc(
        y=audio_segment, sr=int(info["sample_rate"]), n_mfcc=n_mfcc, n_fft=n_fft
    )

    return mfccs

    
def process_single_file(audio_file, output_path, params):
    mfccs = extract_mffc_features(
        audio_file,
        0,
        -1,  # Process entire file
        params["feature_extraction"]["n_mfcc"],
        params["feature_extraction"]["n_fft"]
    )
    # Save MFCC features with same name but .npy extension
    output_file = output_path / (audio_file.stem + '.npy')
    np.save(output_file, mfccs)


def process_audio_data(
    params, input_path: Path, output_path: Path, audio_base_path: Path = AUDIO_DATA_PATH
):
    audio_data = input_path.glob("*.wav")

    # Convert audio files to chunks for feature extraction
    # Get list of all audio files first
    audio_files = list(audio_data)

    process_single_file_partial = functools.partial(
        process_single_file, output_path=output_path, params=params
    )

    # Use multiprocessing with tqdm progress bar
    with Pool() as pool:
        list(tqdm.tqdm(
            pool.map(process_single_file_partial, audio_files),
            total=len(audio_files),
            desc="Extracting MFCC features"
        ))

if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    process_audio_data(
        params, AUDIO_DATA_PATH, Path("./data/mffc_data")
    )
