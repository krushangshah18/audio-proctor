"""
Step 2 — Speaker Diarization (How Many People Are Speaking?)
Detects the number of speakers, timestamps, and creates separate audio files for each.

Features:
- Pitch analysis using autocorrelation
- Energy (RMS) patterns
- Timbre analysis (MFCCs)
- K-means clustering for speaker identification
- Overlap detection (simultaneous speech)
- Outputs separate audio files per speaker + timestamps report
"""

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from scipy import signal
import time
import sys
import os
from collections import defaultdict
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import librosa
import warnings
warnings.filterwarnings('ignore')

# Import Silero VAD
from silero_vad import load_silero_vad, get_speech_timestamps

# Settings
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
ENERGY_THRESHOLD = 0.02
HOP_LENGTH = 512  # For feature extraction
N_MFCC = 13  # Number of MFCC coefficients

# Load models
print("Loading Silero VAD model...")
vad_model = load_silero_vad()

# Buffers
recorded_audio = []
speech_segments = []
speaker_data = defaultdict(lambda: {'samples': [], 'features': [], 'timestamps': []})

start_time = time.perf_counter()

def extract_pitch(audio, sr=SAMPLE_RATE):
    """
    Extract fundamental frequency (pitch) using autocorrelation.
    Returns pitch in Hz or 0 if no clear pitch detected.
    """
    if len(audio) < sr // 40:  # Minimum 25ms
        return 0
    
    # Use librosa's piptrack for pitch estimation
    try:
        # Generate spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=HOP_LENGTH)
        # Get the fundamental frequency
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
        if len(f0) > 0:
            # Get the most common non-zero pitch
            nonzero_pitches = f0[f0 > 0]
            if len(nonzero_pitches) > 0:
                return np.median(nonzero_pitches)
    except Exception as e:
        pass
    
    return 0

def extract_energy(audio):
    """Extract RMS energy"""
    return np.sqrt(np.mean(audio**2))

def extract_mfcc(audio, sr=SAMPLE_RATE):
    """Extract MFCC (Mel-Frequency Cepstral Coefficients) features"""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        # Return mean and std of each MFCC coefficient
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    except:
        return np.zeros(N_MFCC * 2)

def extract_spectral_features(audio, sr=SAMPLE_RATE):
    """Extract spectral centroid and rolloff"""
    try:
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
        return np.array([np.mean(spec_cent), np.std(spec_cent), 
                        np.mean(spec_rolloff), np.std(spec_rolloff)])
    except:
        return np.zeros(4)

def extract_features(audio, sr=SAMPLE_RATE):
    """
    Extract all audio features for a given audio segment.
    Returns feature vector for speaker identification.
    """
    features = {}
    
    # Pitch
    features['pitch'] = extract_pitch(audio, sr)
    
    # Energy
    features['energy'] = extract_energy(audio)
    
    # MFCC
    mfcc_features = extract_mfcc(audio, sr)
    features['mfcc'] = mfcc_features
    
    # Spectral features
    spectral_features = extract_spectral_features(audio, sr)
    features['spectral'] = spectral_features
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    features['zcr'] = zcr
    
    # Combine all features into single vector
    feature_vector = np.concatenate([
        [features['pitch']],
        [features['energy']],
        [features['zcr']],
        features['mfcc'],
        features['spectral']
    ])
    
    return feature_vector, features

def process_audio_file(file_path):
    """Process a pre-recorded audio file with speaker diarization"""
    global recorded_audio, speech_segments
    
    print(f"\nLoading audio file: {file_path}")
    
    # Load audio file
    try:
        sample_rate, audio_data = read(file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return False
    
    # Normalize to float32 [-1, 1]
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
        from scipy import signal as sp_signal
        num_samples = int(len(audio_data) * SAMPLE_RATE / sample_rate)
        audio_data = sp_signal.resample(audio_data, num_samples)
    
    recorded_audio = audio_data
    
    # Get speech segments using VAD
    print("Detecting speech segments with Silero VAD...")
    speech_timestamps = get_speech_timestamps(
        audio_data,
        vad_model,
        sampling_rate=SAMPLE_RATE
    )
    
    speech_segments = speech_timestamps
    print(f"Found {len(speech_segments)} speech segments")
    
    return True

def process_microphone():
    """Record from microphone"""
    global recorded_audio, start_time
    
    print("🎤 Recording... Press Ctrl+C to stop")
    start_time = time.perf_counter()
    
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
    
    # Record until user stops
    frames = []
    try:
        def audio_callback(indata, frames_count, time_info, status):
            frames.append(indata[:, 0].copy())
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SIZE,
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Concatenate all frames
    if frames:
        recorded_audio = np.concatenate(frames)
        
        # Save raw recording
        audio_int16 = (recorded_audio * 32767).astype(np.int16)
        wav_write("_temp_recording.wav", SAMPLE_RATE, audio_int16)
        
        # Get speech segments
        print("Detecting speech segments with Silero VAD...")
        global speech_segments
        speech_segments = get_speech_timestamps(
            recorded_audio,
            vad_model,
            sampling_rate=SAMPLE_RATE
        )
        print(f"Found {len(speech_segments)} speech segments")

def detect_overlaps(speech_segments):
    """
    Detect overlapping speech segments.
    Returns list of overlap regions.
    """
    overlaps = []
    for i, seg1 in enumerate(speech_segments):
        for seg2 in speech_segments[i+1:]:
            start1, end1 = seg1['start'], seg1['end']
            start2, end2 = seg2['start'], seg2['end']
            
            # Check if segments overlap
            if start1 < end2 and start2 < end1:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlaps.append({
                    'start': overlap_start,
                    'end': overlap_end,
                    'duration': overlap_end - overlap_start
                })
    
    return overlaps

def perform_diarization(audio_data, speech_segments):
    """
    Perform speaker diarization using clustering.
    Returns speaker labels for each segment.
    """
    if len(speech_segments) == 0:
        # No speech detected
        print("Warning: No speech segments detected by VAD")
        return [], np.array([])
    
    print("\nExtracting audio features for each segment...")
    print(f"Speech segments found: {len(speech_segments)}")
    print(f"Total audio length: {len(audio_data)} samples ({len(audio_data)/SAMPLE_RATE:.2f}s)")
    segment_features = []
    valid_segments = []
    
    for seg_idx, segment in enumerate(speech_segments):
        start_sample = int(segment['start'])
        end_sample = int(segment['end'])
        
        print(f"  Segment {seg_idx + 1}: {start_sample}-{end_sample} samples ({start_sample/SAMPLE_RATE:.2f}s-{end_sample/SAMPLE_RATE:.2f}s)")
        
        if start_sample >= len(audio_data) or end_sample > len(audio_data):
            print(f"    → Out of bounds, skipping")
            continue
        
        segment_audio = audio_data[start_sample:end_sample]
        seg_duration = len(segment_audio) / SAMPLE_RATE
        
        # Skip very short segments (less than 50ms)
        if len(segment_audio) < SAMPLE_RATE * 0.05:
            print(f"    → Too short ({seg_duration:.3f}s), skipping")
            continue
        
        print(f"    → Processing ({seg_duration:.3f}s)")
        
        try:
            features, feature_dict = extract_features(segment_audio)
            segment_features.append(features)
            valid_segments.append({
                'segment': segment,
                'features': feature_dict,
                'audio': segment_audio,
                'start': segment['start'] / SAMPLE_RATE,
                'end': segment['end'] / SAMPLE_RATE
            })
        except Exception as e:
            print(f"    → Error extracting features: {e}")
            continue
    
    if len(valid_segments) == 0:
        print("No valid segments after filtering")
        return [], np.array([])
    
    if len(valid_segments) < 2:
        # Only 1 speaker or not enough segments
        print(f"Only {len(valid_segments)} valid segment(s) detected")
        return valid_segments, np.array([0])
    
    # Standardize features
    print(f"\nProcessing {len(valid_segments)} segments for speaker clustering...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(segment_features)
    
    # Determine optimal number of speakers
    max_speakers = min(5, len(valid_segments))
    
    # If we have very few segments, limit to fewer clusters
    if len(valid_segments) <= 4:
        n_speakers = min(2, len(valid_segments))
    else:
        # Use simple heuristic: check variance in pitch and energy
        pitches = np.array([v['features']['pitch'] for v in valid_segments])
        energies = np.array([v['features']['energy'] for v in valid_segments])
        
        pitch_variance = np.var(pitches[pitches > 0]) if np.sum(pitches > 0) > 1 else 0
        energy_variance = np.var(energies)
        
        # If high variance, likely multiple speakers
        if pitch_variance > 5000 or energy_variance > 0.001:
            n_speakers = min(3, max_speakers)
        else:
            n_speakers = 2
    
    try:
        print(f"Clustering with n_speakers={n_speakers}...")
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        speaker_labels = kmeans.fit_predict(features_scaled)
        unique_speakers = len(np.unique(speaker_labels))
        print(f"Clustering complete. Detected {unique_speakers} speaker(s)")
    except Exception as e:
        print(f"Clustering error: {e}, defaulting to 1 speaker")
        speaker_labels = np.zeros(len(valid_segments), dtype=int)
    
    return valid_segments, speaker_labels
    
    print("\nExtracting audio features for each segment...")
    segment_features = []
    valid_segments = []
    
    for segment in speech_segments:
        start_sample = int(segment['start'])
        end_sample = int(segment['end'])
        
        if start_sample >= len(audio_data) or end_sample > len(audio_data):
            continue
        
        segment_audio = audio_data[start_sample:end_sample]
        
        # Skip very short segments (less than 50ms)
        if len(segment_audio) < SAMPLE_RATE * 0.05:
            print(f"  Skipped segment {len(valid_segments)+1}: too short ({len(segment_audio)} samples)")
            continue
        
        try:
            features, feature_dict = extract_features(segment_audio)
            segment_features.append(features)
            valid_segments.append({
                'segment': segment,
                'features': feature_dict,
                'audio': segment_audio,
                'start': segment['start'] / SAMPLE_RATE,
                'end': segment['end'] / SAMPLE_RATE
            })
        except Exception as e:
            print(f"Warning: Error extracting features: {e}")
            continue
    
    if len(valid_segments) == 0:
        print("Warning: No valid segments after filtering")
        return [], np.array([])
    
    if len(valid_segments) < 2:
        # Only 1 speaker or not enough segments
        return valid_segments, np.array([0])
    
    # Standardize features
    print(f"Processing {len(valid_segments)} segments for speaker identification...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(segment_features)
    
    # Determine optimal number of speakers using elbow method
    # Start with assumption of 1-5 speakers
    max_speakers = min(5, len(valid_segments))
    
    print(f"Clustering speakers (testing {max_speakers} possible speakers)...")
    
    # If we have very few segments, limit to fewer clusters
    if len(valid_segments) <= 4:
        n_speakers = min(2, len(valid_segments))  # At least try 2, max limited by segment count
    else:
        # Use simple heuristic: check variance in pitch and energy
        pitches = np.array([v['features']['pitch'] for v in valid_segments])
        energies = np.array([v['features']['energy'] for v in valid_segments])
        
        pitch_variance = np.var(pitches[pitches > 0]) if np.sum(pitches > 0) > 1 else 0
        energy_variance = np.var(energies)
        
        # If high variance, likely multiple speakers
        if pitch_variance > 5000 or energy_variance > 0.001:
            n_speakers = min(3, max_speakers)
        else:
            n_speakers = 2  # Default to checking for 2 speakers
    
    try:
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        speaker_labels = kmeans.fit_predict(features_scaled)
    except Exception as e:
        print(f"Clustering error: {e}, defaulting to 1 speaker")
        speaker_labels = np.zeros(len(valid_segments), dtype=int)
    
    return valid_segments, speaker_labels

def main():
    # Check if a file path was provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            sys.exit(1)
        
        if not process_audio_file(input_file):
            sys.exit(1)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
    else:
        print("No input file provided. Recording from microphone...")
        print("Usage: python speaker_diarization.py <audio_file.wav>")
        process_microphone()
        base_name = "recorded"
    
    if isinstance(recorded_audio, list):
        audio_np = np.concatenate(recorded_audio)
    else:
        audio_np = recorded_audio
    
    # Detect overlaps
    print("\nAnalyzing for simultaneous speech (overlaps)...")
    overlaps = detect_overlaps(speech_segments)
    if overlaps:
        print(f"⚠️  Detected {len(overlaps)} overlapping speech regions (multiple speakers detected!)")
    
    # Perform speaker diarization
    valid_segments, speaker_labels = perform_diarization(audio_np, speech_segments)
    
    # Handle case with no valid segments
    if len(valid_segments) == 0:
        print(f"\n{'='*60}")
        print(f"SPEAKER DIARIZATION RESULTS")
        print(f"{'='*60}")
        print("⚠️  No speech detected by VAD. Cannot perform diarization.")
        print(f"{'='*60}\n")
        return
    
    # Organize segments by speaker
    speakers = defaultdict(list)
    all_speaker_audio = defaultdict(list)
    
    num_speakers = len(np.unique(speaker_labels))
    
    print(f"\n{'='*60}")
    print(f"SPEAKER DIARIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Total speakers detected: {num_speakers}")
    print(f"Total speech segments: {len(valid_segments)}")
    print()
    
    for idx, (segment_info, speaker_id) in enumerate(zip(valid_segments, speaker_labels)):
        segment = segment_info['segment']
        start_time = segment_info['start']
        end_time = segment_info['end']
        duration = end_time - start_time
        
        speakers[speaker_id].append({
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'segment_idx': idx
        })
        
        all_speaker_audio[speaker_id].append(segment_info['audio'])
        
        print(f"Segment {idx + 1}: Speaker {speaker_id} | {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
    
    # Save audio files and timestamps for each speaker
    print(f"\n{'='*60}")
    print(f"SAVING OUTPUTS")
    print(f"{'='*60}\n")
    
    timestamps_data = {
        'total_speakers': num_speakers,
        'total_segments': len(valid_segments),
        'overlaps': overlaps,
        'speakers': {}
    }
    
    for speaker_id in sorted(speakers.keys()):
        segments = speakers[speaker_id]
        speaker_audio = np.concatenate(all_speaker_audio[speaker_id])
        
        total_duration = sum(s['duration'] for s in segments)
        
        # Save speaker audio
        audio_int16 = (speaker_audio * 32767).astype(np.int16)
        output_file = f"{base_name}_speaker_{speaker_id}.wav"
        write(output_file, SAMPLE_RATE, audio_int16)
        print(f"Saved Speaker {speaker_id} audio → {output_file}")
        
        # Store in timestamps data
        timestamps_data['speakers'][f'speaker_{speaker_id}'] = {
            'segments': segments,
            'total_duration': total_duration,
            'num_segments': len(segments)
        }
    
    # Save detailed timestamps report
    report_file = f"{base_name}_diarization_report.json"
    with open(report_file, 'w') as f:
        json.dump(timestamps_data, f, indent=2)
    print(f"\nSaved detailed report → {report_file}")
    
    # Save human-readable report
    txt_report_file = f"{base_name}_diarization_report.txt"
    with open(txt_report_file, 'w') as f:
        f.write("SPEAKER DIARIZATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input file: {sys.argv[1] if len(sys.argv) > 1 else 'microphone'}\n")
        f.write(f"Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"Total speakers detected: {num_speakers}\n")
        f.write(f"Total speech segments: {len(valid_segments)}\n\n")
        
        if overlaps:
            f.write("OVERLAPPING SPEECH (Multiple Speakers at Same Time):\n")
            for overlap in overlaps:
                f.write(f"  {overlap['start']/SAMPLE_RATE:.2f}s - {overlap['end']/SAMPLE_RATE:.2f}s " +
                       f"(duration: {overlap['duration']/SAMPLE_RATE:.2f}s)\n")
            f.write("\n")
        
        for speaker_id in sorted(speakers.keys()):
            segments = speakers[speaker_id]
            total_duration = sum(s['duration'] for s in segments)
            
            f.write(f"\nSPEAKER {speaker_id}:\n")
            f.write(f"  Total segments: {len(segments)}\n")
            f.write(f"  Total duration: {total_duration:.2f}s\n")
            f.write(f"  Segments:\n")
            
            for seg in segments:
                f.write(f"    {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        if num_speakers == 1:
            f.write("✓ SINGLE SPEAKER DETECTED - No anomalies\n")
        else:
            f.write(f"⚠️  MULTIPLE SPEAKERS DETECTED - Potential cheating detected!\n")
            if overlaps:
                f.write(f"⚠️  {len(overlaps)} overlapping speech regions detected\n")
    
    print(f"Saved text report → {txt_report_file}")
    
    print(f"\n{'='*60}")
    if num_speakers == 1:
        print(f"✓ RESULT: Single speaker (No anomalies detected)")
    else:
        print(f"⚠️  RESULT: Multiple speakers detected!")
        print(f"Potential cheating detected - {num_speakers} distinct voices found")
        if overlaps:
            print(f"Simultaneous speech detected in {len(overlaps)} region(s)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
