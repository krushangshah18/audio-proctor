"""
Step 1 — Voice Activity Detection (Speech vs Noise)
"""

#Co-pilot CODE
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

# Settings
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
recorded_audio = []

def audio_callback(indata, frames, time_info, status):
    # sounddevice provides float32 samples in range [-1, 1]
    recorded_audio.append(indata[:, 0].copy())

print("🎤 Recording... Press Ctrl+C to stop")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping...")

# Save audio
if recorded_audio:
    audio_np = np.concatenate(recorded_audio)
    # convert float32 [-1,1] to int16
    audio_int16 = (audio_np * 32767).astype(np.int16)
    write("recorded_audio.wav", SAMPLE_RATE, audio_int16)
    print("Saved recording → recorded_audio.wav")
else:
    print("No audio recorded.")
