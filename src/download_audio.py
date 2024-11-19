import os
import argparse
from pathlib import Path
import pandas as pd
import requests
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Download audio files.')
    parser.add_argument('--target-dir', type=str, default='data/raw', help='Target directory for downloaded audio files')
    parser.add_argument('--annotations-file', type=str, default='data/annotations_per_line.csv', help='CSV file with annotations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    target_dir = Path(args.target_dir)
    annotations_file = Path(args.annotations_file)

    # Load the data
    data = pd.read_csv(annotations_file)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Download the audio files
    for i, row in tqdm.tqdm(data.iterrows(), desc='Downloading audio files', total=len(data.index)):
        audio_file_target_path = target_dir / row['audio_path'].split('/')[-1]

        if audio_file_target_path.exists():
            continue
        
        url = f'{os.getenv("LABEL_STUDIO_URL")}{row["audio_path"]}'
        response = requests.get(url, headers={'Authorization': f'Token {os.getenv("API_KEY")}'})
        with audio_file_target_path.open('wb') as f:
            f.write(response.content)
