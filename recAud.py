"""
Step 1 — Voice Activity Detection (Speech vs Noise)

this is basic code to just record audio

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
    write("multiplePeopletest.wav", SAMPLE_RATE, audio_int16)
    print("Saved recording → multiplePeopletest.wav")
else:
    print("No audio recorded.")

"""
My name is [your full name], and today I am taking this exam on my own without assistance. 
I confirm that I will follow all exam rules and maintain academic integrity.

The quick brown fox jumps over the lazy dog. Artificial intelligence and machine learning are 
transforming the world through technology and innovation.

Today’s date is [today’s date], and the time is [current time]. I am ready to begin the exam.

Numbers check: one, three, seven, twelve, twenty-nine, forty-five, and ninety-eight
"""