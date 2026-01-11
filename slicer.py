#!/usr/bin/env python3
"""
Simple Sample Slicer - Cut audio at onsets, group by energy.

Usage:
    python slicer.py track.wav                    # Slice and group
    python slicer.py track.wav --min-dur 2        # Min 2 second segments
    python slicer.py track.wav --groups 4         # 4 energy groups
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("Error: librosa required. Install with: pip install librosa")
    sys.exit(1)


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """Load audio file, return (samples, sample_rate). Always returns stereo."""
    y, sr = librosa.load(filepath, sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # Mono to stereo
    return y, sr


def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """Detect onset times in seconds."""
    mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
    onset_frames = librosa.onset.onset_detect(y=mono, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio segment."""
    return float(np.sqrt(np.mean(audio ** 2)))


def find_cut_points(onset_times: np.ndarray, duration: float,
                    min_gap: float = 0.3) -> list[float]:
    """Find cut points at significant gaps between onsets."""
    if len(onset_times) < 2:
        return [0.0, duration]

    # Find gaps larger than min_gap
    gaps = np.diff(onset_times)
    gap_indices = np.where(gaps > min_gap)[0]

    # Cut points are at the onset AFTER each significant gap
    cut_points = [0.0]
    for idx in gap_indices:
        cut_points.append(onset_times[idx + 1])
    cut_points.append(duration)

    return cut_points


def slice_at_gaps(audio: np.ndarray, sr: int, onset_times: np.ndarray,
                  min_dur: float = 1.0, max_dur: float = 10.0,
                  min_gap: float = 0.3, min_rms: float = 0.01) -> list[dict]:
    """Slice audio at significant gaps, filter by duration and RMS."""
    duration = audio.shape[1] / sr

    # Find cut points at gaps
    cut_points = find_cut_points(onset_times, duration, min_gap)
    print(f"  Found {len(cut_points) - 1} potential segments (gap > {min_gap}s)")

    segments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        seg_dur = end - start

        # Split long segments at max_dur boundaries
        while seg_dur > max_dur:
            # Take a max_dur chunk
            chunk_end = start + max_dur
            seg = _extract_segment(audio, sr, start, chunk_end)
            if seg["rms"] >= min_rms:
                segments.append(seg)
            start = chunk_end
            seg_dur = end - start

        # Keep remaining if >= min_dur
        if seg_dur >= min_dur:
            seg = _extract_segment(audio, sr, start, end)
            if seg["rms"] >= min_rms:
                segments.append(seg)

    return segments


def _extract_segment(audio: np.ndarray, sr: int, start: float, end: float) -> dict:
    """Extract a segment and compute its features."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment_audio = audio[:, start_sample:end_sample]
    rms = calculate_rms(segment_audio)

    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "rms": rms,
        "audio": segment_audio,
    }


def group_by_energy(segments: list[dict], num_groups: int = 4) -> dict[int, list[dict]]:
    """Group segments by energy using K-means clustering."""
    from sklearn.cluster import KMeans

    if not segments:
        return {}

    if len(segments) <= num_groups:
        # Not enough segments to cluster meaningfully
        return {i: [seg] for i, seg in enumerate(segments)}

    # Extract RMS values as features
    rms_values = np.array([[seg["rms"]] for seg in segments])

    # K-means clustering
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rms_values)

    # Get cluster centers and sort by energy (so group 0 = quietest)
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    label_map = {old: new for new, old in enumerate(sorted_indices)}

    # Group segments by remapped cluster label
    groups = {}
    for seg, label in zip(segments, labels):
        new_label = label_map[label]
        if new_label not in groups:
            groups[new_label] = []
        groups[new_label].append(seg)

    # Sort segments within each group by RMS
    for group_idx in groups:
        groups[group_idx].sort(key=lambda x: x["rms"])

    return groups


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def export_segments(groups: dict[int, list[dict]], sr: int, output_dir: Path):
    """Export segments organized by group."""
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_idx, segments in groups.items():
        group_dir = output_dir / f"group_{group_idx}"
        group_dir.mkdir(exist_ok=True)

        for i, seg in enumerate(segments):
            filename = f"seg_{i:03d}_{seg['duration']:.1f}s.wav"
            filepath = group_dir / filename

            # Transpose to (samples, channels) for soundfile
            audio_out = seg["audio"].T
            sf.write(filepath, audio_out, sr)


def main():
    parser = argparse.ArgumentParser(
        description="Slice audio at onsets, group by energy."
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--min-dur", type=float, default=1.0,
                        help="Minimum segment duration in seconds (default: 1.0)")
    parser.add_argument("--max-dur", type=float, default=10.0,
                        help="Maximum segment duration in seconds (default: 10.0)")
    parser.add_argument("--gap", type=float, default=0.3,
                        help="Minimum gap between onsets to cut at (default: 0.3)")
    parser.add_argument("--groups", type=int, default=4,
                        help="Number of energy groups (default: 4)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: ./slices/<filename>)")

    args = parser.parse_args()

    # Load audio
    print(f"Loading {args.input}...")
    audio, sr = load_audio(args.input)
    duration = audio.shape[1] / sr
    print(f"  Duration: {format_time(duration)} ({sr}Hz)")

    # Detect onsets
    print("Detecting onsets...")
    onset_times = detect_onsets(audio, sr)
    print(f"  Found {len(onset_times)} onsets")

    # Slice at gaps
    print(f"Slicing (keeping {args.min_dur}-{args.max_dur}s segments)...")
    segments = slice_at_gaps(audio, sr, onset_times,
                             min_dur=args.min_dur, max_dur=args.max_dur,
                             min_gap=args.gap)
    print(f"  Created {len(segments)} segments")

    if not segments:
        print("No segments found. Try adjusting --min-dur or --max-dur")
        sys.exit(1)

    # Group by energy
    print(f"Grouping into {args.groups} energy levels...")
    groups = group_by_energy(segments, num_groups=args.groups)

    # Report
    print("\nSegments:")
    print("-" * 60)
    for i, seg in enumerate(segments):
        print(f"  {i+1:2d}. {format_time(seg['start'])} - {format_time(seg['end'])} "
              f"({seg['duration']:.1f}s, RMS: {seg['rms']:.4f})")
    print("-" * 60)

    print("\nGroups (by energy, low to high):")
    print("-" * 60)
    for group_idx in sorted(groups.keys()):
        segs = groups[group_idx]
        avg_rms = np.mean([s["rms"] for s in segs])
        total_dur = sum(s["duration"] for s in segs)
        print(f"  Group {group_idx}: {len(segs):3d} segments, "
              f"{total_dur:5.1f}s total, avg RMS: {avg_rms:.4f}")
        for seg in segs:
            print(f"         {format_time(seg['start'])} - {format_time(seg['end'])} ({seg['duration']:.1f}s)")
    print("-" * 60)

    # Export
    input_stem = Path(args.input).stem
    output_dir = Path(args.output) if args.output else Path("./slices") / input_stem

    print(f"\nExporting to {output_dir}/")
    export_segments(groups, sr, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
