import pyaudio
import wave
import numpy as np
import collections
import datetime
import time
import argparse
import os
from pathlib import Path
import xgboost as xgb
import librosa

# Modified parameters for Pi Zero W
RATE = 16000
WINDOW_SIZE = 0.1
CHANNELS = 1
FORMAT = pyaudio.paInt16  # Use 16-bit instead of float32 to reduce memory
CHUNK = int(RATE * WINDOW_SIZE)


# Modified save_audio function
def save_audio(initial_data, stream, filename, duration=7):
    frames = list(initial_data)
    chunks_to_record = int((RATE * duration) / CHUNK)

    for _ in range(chunks_to_record):
        data = stream.read(
            CHUNK, exception_on_overflow=False
        )  # Handle potential overflows
        frames.append(data)

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for Int16
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def parse_args():
    parser = argparse.ArgumentParser(description="Audio monitoring and recording")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="recordings",
        help="Directory to save recordings (default: recordings)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold of the model between 0 and 1")
    parser.add_argument("--duration", type=int, default=7)
    parser.add_argument(
        "--keep-chunks",
        type=int,
        default=5,
        help="Number of previous windows to keep in memory",
    )
    parser.add_argument(
        "--skip-chunks",
        type=int,
        default=10,
        help="Number of chunks to skip before starting to record",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the XGBoost model file",
    )
    return parser.parse_args()


# Modified main function
def main():

    args = parse_args()

    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    p = pyaudio.PyAudio()

    # Search for the microphone device index
    device_idx = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if (
            "seeed-2mic-voicecard" in device_info["name"]
            and device_info["maxInputChannels"] > 0
        ):
            device_idx = i
            break

    if device_idx is None:
        print("Microphone device not found")
        return

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_idx,
    )

    # Load XGBoost model from JSON file

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    model = xgb.Booster()
    model.load_model(model_path)
    print(f"Loaded XGBoost model from {model_path}")

    # Create circular buffer for storing audio history
    audio_buffer = collections.deque(maxlen=args.keep_chunks)

    print(f"Monitoring audio at 16kHz... Saving to {save_dir}/")

    i = 0  # Counter for printing mean amplitude from time to time
    skip_chunks = args.skip_chunks
    n_mfcc = 13
    n_fft = 512
    chunk_rotation_idx = 0
    try:
        while True:
            try:
                data = stream.read(CHUNK)

                if skip_chunks > 0:
                    skip_chunks -= 1
                    continue

                audio_buffer.append(data)

                # Only process every 3rd chunk to reduce CPU usage
                if chunk_rotation_idx % 3 == 0:
                    chunk_rotation_idx = 1
                    continue
                chunk_rotation_idx += 1

                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_data = np.float32(audio_data)
                mfccs = librosa.feature.mfcc(
                    y=audio_data, sr=RATE, n_mfcc=n_mfcc, n_fft=n_fft
                )
                X = mfccs.flatten()
                X = xgb.DMatrix(X.reshape((1, X.shape[0])))

                y_pred = model.predict(X)
                if y_pred[0] > args.threshold:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = save_dir / f"recording_{timestamp}.wav"
                    print(f"[{timestamp}] Threshold exceeded with {y_pred}! Recording to {filename}")
                    save_audio(audio_buffer, stream, filename)
                # mean_amplitude = np.mean(np.abs(audio_data))
                # if i % 100 == 0:
                #     print(f"Still running, current mean amplitude: {mean_amplitude}")
                #     i = 0
                # if mean_amplitude > args.threshold:
                #     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                #     filename = save_dir / f"recording_{timestamp}.wav"
                #     print(
                #         f"[{timestamp}] Threshold exceeded with {mean_amplitude}! Recording to {filename}"
                #     )
                #     save_audio(audio_buffer, stream, filename)
                # i += 1

            except OSError:
                print("OSError: Buffer overflow, restart")
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_idx,
                )
                skip_chunks = args.skip_chunks

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
