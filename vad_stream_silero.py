"""
Step 1 — Voice Activity Detection (Speech vs Noise) using Silero VAD
Can process pre-recorded audio file or record from microphone
"""
import sounddevice as sd #capture microphone audio in real time
import numpy as np
from scipy.io.wavfile import write, read #save and load audio from a WAV file
import time
import noisereduce as nr  # for noise reduction
import sys
import os

# Import Silero VAD
from silero_vad import load_silero_vad, get_speech_timestamps

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

# initialize Silero VAD
print("Loading Silero VAD model...")
model = load_silero_vad()

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

    # Silero VAD processing - works directly with float32 audio [-1, 1]
    # Note: In streaming mode, we process per-frame, but Silero VAD works better
    # on longer segments. In file mode, we'll use get_speech_timestamps instead.

    print(f"{timestamp:.2f}s – energy: {energy:.4f}")

    if speech_active:
        speech_audio.append(audio)

def process_audio_file(file_path):
    """Process a pre-recorded audio file with Silero VAD"""
    global recorded_audio, speech_audio, speech_segments
    
    print(f"\nLoading audio file: {file_path}")
    
    # Load audio file
    try:
        sample_rate, audio_data = read(file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    # Normalize to float32 [-1, 1] if it's int16
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32767.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483647.0
    
    # Handle stereo - convert to mono if needed
    if len(audio_data.shape) > 1:
        print(f"Converting {audio_data.shape[1]} channels to mono...")
        audio_data = audio_data[:, 0]
    
    # Resample if necessary
    if sample_rate != SAMPLE_RATE:
        print(f"Resampling from {sample_rate}Hz to {SAMPLE_RATE}Hz...")
        from scipy import signal
        num_samples = int(len(audio_data) * SAMPLE_RATE / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
    
    recorded_audio = [audio_data]  # Store full audio
    
    # Get speech timestamps using Silero VAD
    print("Detecting speech segments with Silero VAD...")
    speech_timestamps = get_speech_timestamps(
        audio_data,
        model,
        sampling_rate=SAMPLE_RATE
    )
    
    print(f"\nFound {len(speech_timestamps)} speech segments:")
    
    # Process each speech segment
    speech_segments = []
    for segment in speech_timestamps:
        start_sample = segment['start']
        end_sample = segment['end']
        start_time_sec = start_sample / SAMPLE_RATE
        end_time_sec = end_sample / SAMPLE_RATE
        
        print(f"  {start_time_sec:.2f}s → {end_time_sec:.2f}s (duration: {(end_time_sec - start_time_sec):.2f}s)")
        speech_segments.append((start_time_sec, end_time_sec))
        
        # Extract speech audio for this segment
        speech_audio.append(audio_data[start_sample:end_sample])
    
    return True

def process_microphone():
    """Record and process audio from microphone"""
    global start_time
    
    print("🎤 Recording... Press Ctrl+C to stop")
    start_time = time.perf_counter()
    
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

def main():
    # Check if a file path was provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            sys.exit(1)
        process_audio_file(input_file)
    else:
        print("No input file provided. Recording from microphone...")
        print("Usage: python vad_stream_silero.py <audio_file.wav>")
        process_microphone()
    
    # Save audio outputs
    base_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "recorded"
    
    # Save full recording
    if recorded_audio:
        if len(recorded_audio) == 1:
            audio_np = recorded_audio[0]
        else:
            audio_np = np.concatenate(recorded_audio)
        
        # convert float32 [-1,1] to int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        output_file = f"{base_name}_silero_full.wav"
        write(output_file, SAMPLE_RATE, audio_int16)
        print(f"\nSaved full recording → {output_file}")
    else:
        print("\nNo audio recorded.")

    # write speech-only file
    if speech_audio:
        speech_np = np.concatenate(speech_audio)
        # apply noise reduction
        speech_clean = nr.reduce_noise(y=speech_np, sr=SAMPLE_RATE)
        speech_int16 = (speech_clean * 32767).astype(np.int16)
        output_file = f"{base_name}_silero_speech.wav"
        write(output_file, SAMPLE_RATE, speech_int16)
        print(f"Saved speech-only → {output_file}")
        
        # Save a report with timestamps
        report_file = f"{base_name}_silero_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Speech Detection Report (Silero VAD)\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Input file: {sys.argv[1] if len(sys.argv) > 1 else 'microphone'}\n")
            f.write(f"Sample rate: {SAMPLE_RATE} Hz\n")
            f.write(f"\nSpeech Segments:\n")
            for i, (start, end) in enumerate(speech_segments, 1):
                duration = end - start
                f.write(f"  Segment {i}: {start:.2f}s → {end:.2f}s (duration: {duration:.2f}s)\n")
            f.write(f"\nTotal segments: {len(speech_segments)}\n")
            total_speech = sum(end - start for start, end in speech_segments)
            f.write(f"Total speech duration: {total_speech:.2f}s\n")
        print(f"Saved report → {report_file}")
    else:
        print("\nNo speech detected, nothing saved to speech output.")

if __name__ == "__main__":
    main()