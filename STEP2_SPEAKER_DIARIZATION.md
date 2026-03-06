# Step 2: Speaker Diarization - How Many People Are Speaking?

## Overview

Speaker diarization detects the **number of speakers** and **when each speaker talks** in an audio recording. This is critical for exam proctoring to detect potential cheating (multiple voices).

## Features

✅ **Multi-Speaker Detection** - Identifies if 1 or multiple speakers  
✅ **Speaker Timestamps** - Exact timestamps when each speaker talks  
✅ **Speaker Audio Separation** - Extracts separate audio files for each speaker  
✅ **Overlap Detection** - Detects simultaneous speech (multiple speakers at same time)  
✅ **Audio Feature Analysis**:

- Pitch/Frequency analysis using autocorrelation and librosa
- Energy (RMS) patterns
- MFCC (Mel-Frequency Cepstral Coefficients) for timbre
- Spectral features (centroid, rolloff, ZCR)
- K-means clustering for speaker identification

## Installation

All dependencies are already in `pyproject.toml`:

- `librosa>=0.10.0` - Audio feature extraction
- `scikit-learn>=1.3.0` - Clustering algorithms
- `silero-vad>=5.1` - Speech segmentation
- Plus previous dependencies (numpy, scipy, sounddevice, etc.)

## Usage

### Process Pre-Recorded Audio File

```bash
python speaker_diarization.py <audio_file.wav>
```

Example:

```bash
# Process audio from VAD step
python speaker_diarization.py recorded_audio.wav
```

### Record from Microphone (Live)

```bash
python speaker_diarization.py
```

Then press `Ctrl+C` to stop recording.

## Output Files

For input file `audio.wav`, generates:

1. **`audio_speaker_0.wav`** - Speaker 0 audio
2. **`audio_speaker_1.wav`** - Speaker 1 audio (if multiple speakers)
3. **`audio_speaker_N.wav`** - Speaker N audio
4. **`audio_diarization_report.json`** - Detailed timestamps in JSON format
5. **`audio_diarization_report.txt`** - Human-readable report

## Output Examples

### Single Speaker Result

```
============================================================
SPEAKER DIARIZATION RESULTS
============================================================
Total speakers detected: 1
Total speech segments: 1

Segment 1: Speaker 0 | 1.54s - 7.36s (5.82s)

============================================================
✓ RESULT: Single speaker (No anomalies detected)
============================================================
```

### Multi-Speaker Result

```
============================================================
SPEAKER DIARIZATION RESULTS
============================================================
Total speakers detected: 2
Total speech segments: 4

Segment 1: Speaker 1 | 0.96s - 5.15s (4.19s)
Segment 2: Speaker 0 | 6.69s - 7.42s (0.73s)
Segment 3: Speaker 0 | 7.65s - 8.22s (0.57s)
Segment 4: Speaker 0 | 8.42s - 10.00s (1.58s)

============================================================
⚠️  RESULT: Multiple speakers detected!
Potential cheating detected - 2 distinct voices found
============================================================
```

## JSON Report Structure

```json
{
  "total_speakers": 2,
  "total_segments": 4,
  "overlaps": [],
  "speakers": {
    "speaker_0": {
      "segments": [
        { "start": 6.69, "end": 7.42, "duration": 0.73, "segment_idx": 1 },
        { "start": 7.65, "end": 8.22, "duration": 0.57, "segment_idx": 2 },
        { "start": 8.42, "end": 10.0, "duration": 1.58, "segment_idx": 3 }
      ],
      "total_duration": 3.88,
      "num_segments": 3
    },
    "speaker_1": {
      "segments": [
        { "start": 0.96, "end": 5.15, "duration": 4.19, "segment_idx": 0 }
      ],
      "total_duration": 4.19,
      "num_segments": 1
    }
  }
}
```

## Testing

Generate synthetic test audio:

```bash
python generate_test_audio.py
```

This creates:

- **`test_single_speaker.wav`** - Single speaker (1-9s) with noise
- **`test_multi_speaker.wav`** - Two speakers with overlaps (3-4s, 7-8s)

Test them:

```bash
python speaker_diarization.py test_single_speaker.wav
python speaker_diarization.py test_multi_speaker.wav
```

## Algorithm Details

### Step 1: Speech Segmentation

Uses Silero VAD (from Step 1) to detect speech segments and silence periods.

### Step 2: Feature Extraction (Per Segment)

For each speech segment, extracts:

- **Pitch** - Fundamental frequency using autocorrelation
- **Energy** - RMS energy for loudness analysis
- **MFCC** - 13 Mel-Frequency Cepstral Coefficients (timbre)
- **Spectral Features** - Centroid, rolloff, zero-crossing rate
- **Total: 33-dimensional feature vector per segment**

### Step 3: Clustering

- Uses **K-means clustering** to group segments by speaker
- Automatic speaker count detection based on:
  - Pitch variance (high variance → multiple speakers)
  - Energy variance
  - Number of segments
- Default: 2 speakers tested; adjusts up to 5 if high variance detected

### Step 4: Overlap Detection

Checks if any two segments overlap in time, indicating simultaneous speech.

## Limitations & Considerations

⚠️ **Limitations:**

- Works best with **clear speech** and **distinct voices**
- May struggle with:
  - Very similar voices (twins, siblings)
  - Whispered/quiet speech
  - Heavy accents or non-native speakers
  - Too much background noise
- Synthetic audio may not be detected properly by VAD

✅ **Good scenarios:**

- Exam proctor detecting unauthorized third party
- Identifying candidate trying to use AI voice assistance
- Multiple distinct speakers (e.g., male + female)

## Integration with Previous Steps

After **Step 1 (VAD)**:

```bash
# Step 1: Run VAD and save audio
python vad_stream_silero.py recorded_audio.wav

# Step 2: Run speaker diarization on VAD output
python speaker_diarization.py recorded_audio_silero_full.wav
```

## Future Enhancements

Planned improvements:

- [ ] Neural speaker embeddings (d-vector, x-vector) for better accuracy
- [ ] Larger model support (pyannote.audio for production)
- [ ] Browser-based version using Web Audio API
- [ ] Real-time streaming diarization
- [ ] Noise-robust speaker identification
- [ ] Support for >5 speakers
- [ ] Speaker re-identification across sessions

## References

- **Silero VAD**: https://github.com/snakers4/silero-vad
- **Librosa**: https://librosa.org/ - Music and audio analysis
- **Scikit-learn KMeans**: https://scikit-learn.org/
- **Speaker Diarization Papers**:
  - "Speaker Diarization: A Review of Challenges, Recent Advances, and Next Frontiers"
  - "End-to-End Speaker Diarization for an Unknown Number of Speakers"
