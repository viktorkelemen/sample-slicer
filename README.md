# Sample Slicer

Slice audio at onsets, group by energy level.

## Setup

```bash
cd /Users/vkelemen/workspace/sample-slicer
python3 -m venv venv
source venv/bin/activate
pip install librosa soundfile numpy
```

Or use the existing sample-cutter venv:
```bash
source /Users/vkelemen/workspace/sample-cutter/venv/bin/activate
```

## Usage

```bash
python slicer.py <audio_file> [options]
```

### Examples

```bash
# Basic - slice and group into 4 energy levels
python slicer.py ~/Desktop/track.wav

# Adjust segment duration limits (default: 1-10 seconds)
python slicer.py track.wav --min-dur 2 --max-dur 8

# Change number of energy groups (default: 4)
python slicer.py track.wav --groups 6

# Adjust gap detection sensitivity (default: 0.3s)
python slicer.py track.wav --gap 0.5

# Custom output directory
python slicer.py track.wav -o ./my_slices
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min-dur` | 1.0 | Minimum segment duration (seconds) |
| `--max-dur` | 10.0 | Maximum segment duration (seconds) |
| `--gap` | 0.3 | Minimum gap between onsets to cut at (seconds) |
| `--groups` | 4 | Number of energy groups |
| `-o, --output` | `./slices/<filename>` | Output directory |

## Output Structure

```
slices/<track_name>/
├── group_0/   (quietest)
│   ├── seg_000_2.1s.wav
│   └── ...
├── group_1/
├── group_2/
└── group_3/   (loudest)
```

## How It Works

1. **Load audio** - supports WAV, AIFF, M4A, etc. via librosa
2. **Detect onsets** - find transients/attacks in the audio
3. **Find gaps** - cut at significant pauses (gap > threshold)
4. **Filter by duration** - keep segments within min/max range
5. **Group by RMS energy** - sort into energy buckets (quiet → loud)
6. **Export** - save WAV files organized by group

## Next Steps

- [ ] Play samples from SuperCollider with rhythm patterns
- [ ] Live triggering via MIDI/keyboard
