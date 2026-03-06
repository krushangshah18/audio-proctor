"""
Test script for speaker diarization
Generates synthetic multi-speaker audio and tests the diarization
"""
import numpy as np
from scipy.io.wavfile import write
import sys

def generate_synthetic_speech(duration=10):
    """
    Generate synthetic multi-speaker audio with overlapping speech.
    Uses more realistic speech-like characteristics.
    
    Contains:
    - Speaker 1: Low pitch (150-200 Hz base), segments at [1-4s], [6-8s]
    - Speaker 2: High pitch (250-300 Hz base), segments at [3-5s], [7-10s]
    - Overlap: 3-4s (both speakers), 7-8s (both speakers)
    - Noise: Throughout
    """
    SAMPLE_RATE = 16000
    num_samples = duration * SAMPLE_RATE
    t = np.linspace(0, duration, num_samples)
    audio = np.zeros_like(t)
    
    # Helper to create speech-like audio (harmonic series)
    def create_speech_like(freq, start_time, end_time, t, SAMPLE_RATE):
        envelope = np.zeros_like(t)
        envelope[(int(start_time*SAMPLE_RATE)) : (int(end_time*SAMPLE_RATE))] = 1
        
        # Add harmonics for more realistic speech
        signal = np.zeros_like(t)
        for harmonic in range(1, 5):
            h_freq = freq * harmonic
            amplitude = 1.0 / (harmonic ** 1.5)  # Decrease amplitude for higher harmonics
            # Add frequency modulation
            fm = 20 * np.sin(2 * np.pi * 2 * t)  # 2Hz vibrato
            signal += amplitude * np.sin(2 * np.pi * (h_freq + fm) * t)
        
        # Add amplitude modulation for speech-like quality
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3Hz amplitude modulation
        signal = signal * am
        
        return envelope * signal
    
    # Speaker 1: Lower frequency (male-like voice)
    speaker1 = create_speech_like(150, 1.0, 4.0, t, SAMPLE_RATE)
    speaker1 += create_speech_like(160, 6.0, 8.0, t, SAMPLE_RATE)
    
    # Speaker 2: Higher frequency (female-like voice)  
    speaker2 = create_speech_like(280, 3.0, 5.0, t, SAMPLE_RATE)
    speaker2 += create_speech_like(290, 7.0, 10.0, t, SAMPLE_RATE)
    
    # Combine speakers (normalized)
    audio = speaker1 * 0.35 + speaker2 * 0.35
    
    # Add colored noise (more realistic)
    noise = np.random.randn(len(t))
    # Simple low-pass filter for colored noise
    from scipy import signal as sp_signal
    b, a = sp_signal.butter(2, 0.1)  # Low-pass filter
    noise_colored = sp_signal.filtfilt(b, a, noise)
    noise_colored = noise_colored / np.max(np.abs(noise_colored)) * 0.1
    audio += noise_colored
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 0.01) * 0.85
    
    return audio.astype(np.float32), SAMPLE_RATE

def generate_single_speaker(duration=10):
    """Generate synthetic single-speaker audio with more realistic characteristics"""
    SAMPLE_RATE = 16000
    num_samples = duration * SAMPLE_RATE
    t = np.linspace(0, duration, num_samples)
    
    # Create envelope: speaker talks during [1-9s]
    envelope = np.zeros_like(t)
    envelope[int(1*SAMPLE_RATE):int(9*SAMPLE_RATE)] = 1
    
    # Single speaker with consistent base frequency
    freq = 180  # Single person's base frequency
    signal = np.zeros_like(t)
    
    # Add harmonics for realistic speech
    for harmonic in range(1, 6):
        h_freq = freq * harmonic
        amplitude = 1.0 / (harmonic ** 1.5)
        freq_mod = 15 * np.sin(2 * np.pi * 2.5 * t)  # 2.5Hz vibrato
        signal += amplitude * np.sin(2 * np.pi * (h_freq + freq_mod) * t)
    
    # Add amplitude modulation
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)  # 2Hz AM
    signal = signal * am
    
    audio = envelope * signal * 0.4
    
    # Add light colored noise
    from scipy import signal as sp_signal
    noise = np.random.randn(len(t))
    b, a = sp_signal.butter(2, 0.1)
    noise_colored = sp_signal.filtfilt(b, a, noise)
    noise_colored = noise_colored / np.max(np.abs(noise_colored)) * 0.05
    audio += noise_colored
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 0.01) * 0.85
    
    return audio.astype(np.float32), SAMPLE_RATE

def main():
    print("Generating synthetic test audio...\n")
    
    # Generate single speaker audio
    print("1. Single Speaker Test:")
    print("-" * 40)
    print("Generating 10s of single speaker audio...")
    single_audio, sr = generate_single_speaker(10)
    write("test_single_speaker.wav", sr, (single_audio * 32767).astype(np.int16))
    print("Saved: test_single_speaker.wav")
    print()
    
    # Generate multi-speaker audio
    print("2. Multi-Speaker Test:")
    print("-" * 40)
    print("Generating 10s of multi-speaker audio with overlaps...")
    print("  - Speaker 1 (low pitch): 0-3s, 5-7s")
    print("  - Speaker 2 (high pitch): 2-4s, 6-9s")
    print("  - Overlaps: 2-3s, 6-7s")
    multi_audio, sr = generate_synthetic_speech(10)
    write("test_multi_speaker.wav", sr, (multi_audio * 32767).astype(np.int16))
    print("Saved: test_multi_speaker.wav")
    print()
    
    print("=" * 40)
    print("Test files generated!")
    print("=" * 40)
    print("\nNow test speaker diarization:")
    print("\n  Single speaker test:")
    print("  $ python speaker_diarization.py test_single_speaker.wav")
    print("\n  Multi-speaker test:")
    print("  $ python speaker_diarization.py test_multi_speaker.wav")
    print()

if __name__ == "__main__":
    main()
