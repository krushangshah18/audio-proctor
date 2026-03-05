"""
Step 1 — Voice Activity Detection (Speech vs Noise)
"""
import sounddevice as sd #capture microphone audio in real time
import numpy as np
from scipy.io.wavfile import write #save audio to a WAV file
import time

# Settings
"""
Human speech mostly lies in : 85 Hz – 8000 Hz
Nyquist theorem : sample_rate ≥ 2 × max_frequency
"""
SAMPLE_RATE = 16000 # its 16,000 Hz : 16000 audio samples per second

FRAME_DURATION = 30  # ms Audio is processed in small chunks called frames

# Audio is processed in small chunks called frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

# simple energy threshold for noise filtering
ENERGY_THRESHOLD = 0.023

# initialize webrtc VAD
#Google's WebRTC Voice Activity Detector
import webrtcvad
vad = webrtcvad.Vad(3)  # aggressive

# Buffers to store different data
recorded_audio = []
speech_audio = []          # store only frames classified as speech
speech_segments = []

#Used to detect start and end of speech segments
speech_active = False
speech_start = None

start_time = time.perf_counter()

#called every time a new audio frame arrives
def audio_callback(indata, #numpy array containing audio samples Shape : (frames, channels)
                   frames, #Number of samples in this chunk
                   time_info, # Audio timing info from PortAudio
                   status
                   ):
    global speech_active, speech_start

    audio = indata[:, 0].copy()

    #allows us to reconstruct the full recording later
    recorded_audio.append(audio) 

    """
    computes RMS energy
    Formula : RMS = √(mean(x²))
    Higher RMS → stronger signal
    """
    energy = np.sqrt(np.mean(audio**2))
    timestamp = time.perf_counter() - start_time

    if energy < ENERGY_THRESHOLD:
        print(f"{timestamp:.2f}s – noise (energy {energy:.4f})")
        return

    # VAD only accepts 10/20/30‑ms frames; ensure length matches
    if len(audio) != FRAME_SIZE:
        # skip or pad if necessary
        print(f"{timestamp:.2f}s – unexpected frame size {len(audio)}")
        return
    """
    Microphone gives float32 values range = [-1 , 1]
    But VAD expects PCM int16 (PCM range: -32768 → 32767) So we scale.
    """
    pcm = (audio * 32767).astype(np.int16)

    try:
        #WebRTC VAD analyzes the frame : True  → speech, False → noise
        is_speech = vad.is_speech(pcm.tobytes(), SAMPLE_RATE)
    except webrtcvad.Error as e:
        print(f"VAD error: {e}")
        return
    print(f"{timestamp:.2f}s – speech? {is_speech}")

    if is_speech:
        speech_audio.append(audio)

    if is_speech and not speech_active:
        speech_start = timestamp
        speech_active = True
    elif not is_speech and speech_active:
        speech_segments.append((speech_start, timestamp))
        speech_active = False

print("🎤 Recording... Press Ctrl+C to stop")

try:
    with sd.InputStream( # Creates microphone stream
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=FRAME_SIZE,
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

# write speech-only file
if speech_audio:
    speech_np = np.concatenate(speech_audio)
    speech_int16 = (speech_np * 32767).astype(np.int16)
    write("speech_only.wav", SAMPLE_RATE, speech_int16)
    print("Saved speech-only → speech_only.wav")
else:
    print("No speech detected, nothing saved to speech_only.wav")
