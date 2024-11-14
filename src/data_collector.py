import pyaudio
import wave
import numpy as np
import collections
import datetime
import time
import argparse
import os
from pathlib import Path

# Modified parameters for Pi Zero W
RATE = 16000
WINDOW_SIZE = 0.2  # Increase to 200ms to reduce CPU load
CHANNELS = 1
FORMAT = pyaudio.paInt16  # Use 16-bit instead of float32 to reduce memory
CHUNK = int(RATE * WINDOW_SIZE)

# Modified save_audio function
def save_audio(initial_data, stream, filename, duration=7):
    frames = list(initial_data)
    chunks_to_record = int((RATE * duration) / CHUNK)
    
    for _ in range(chunks_to_record):
        data = stream.read(CHUNK, exception_on_overflow=False)  # Handle potential overflows
        frames.append(data)
    
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for Int16
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def parse_args():
    parser = argparse.ArgumentParser(description='Audio monitoring and recording')
    parser.add_argument('--save-dir', type=str, default='recordings',
                       help='Directory to save recordings (default: recordings)')
    parser.add_argument('--threshold', type=int, default=1000)
    parser.add_argument('--duration', type=int, default=7)
    parser.add_argument('--keep', type=int, default=5, help='Number of previous windows to keep in memory')
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
        if 'seeed-2mic-voicecard' in device_info['name'] and device_info['maxInputChannels'] > 0:
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
        input_device_index=device_idx
    )

    
    # Create circular buffer for storing audio history
    audio_buffer = collections.deque(maxlen=args.keep)

    print(f"Monitoring audio at 16kHz... Saving to {save_dir}/")

    try:
        while True:
            data = stream.read(CHUNK)
            audio_buffer.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            mean_amplitude = np.mean(np.abs(audio_data))
            print(f'Mean amplitude: {mean_amplitude}')
            if mean_amplitude > args.threshold:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = save_dir / f"recording_{timestamp}.wav"
                print(f"[{timestamp}] Threshold exceeded! Recording to {filename}")
                save_audio(audio_buffer, stream, filename)
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
