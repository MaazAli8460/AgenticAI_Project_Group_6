from __future__ import annotations

import math
import re
import wave
from array import array
from pathlib import Path
from typing import Iterable

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2
CHANNELS = 1
MAX_INT16 = 32767


def estimate_duration_ms(text: str, wpm: int = 150, min_ms: int = 800) -> int:
    words = len(re.findall(r"\w+", text))
    seconds = words / (wpm / 60.0) if words else 0.0
    seconds = max(seconds, min_ms / 1000.0)
    return int(seconds * 1000)


def generate_tone_samples(
    duration_ms: int,
    frequency_hz: float = 220.0,
    volume: float = 0.2,
) -> array:
    total_samples = int(SAMPLE_RATE * duration_ms / 1000)
    amplitude = int(MAX_INT16 * max(0.0, min(volume, 1.0)))
    samples = array("h")
    for i in range(total_samples):
        value = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * i / SAMPLE_RATE))
        samples.append(value)
    return samples


def generate_silence_samples(duration_ms: int) -> array:
    total_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return array("h", [0] * total_samples)


def pad_samples_to_duration(samples: array, target_duration_ms: int) -> array:
    target_samples = int(SAMPLE_RATE * target_duration_ms / 1000)
    if len(samples) > target_samples:
        return array("h", samples[:target_samples])
    if len(samples) < target_samples:
        samples.extend([0] * (target_samples - len(samples)))
    return samples


def write_wav(path: Path, samples: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def read_wav(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != CHANNELS or wav_file.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("Only 16-bit mono WAV is supported.")
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Unsupported sample rate for audio merge.")
    samples = array("h")
    samples.frombytes(frames)
    return sample_rate, samples


def concat_samples(chunks: Iterable[array], gap_ms: int = 0) -> array:
    output = array("h")
    gap = generate_silence_samples(gap_ms) if gap_ms > 0 else None
    for idx, chunk in enumerate(chunks):
        if idx > 0 and gap is not None:
            output.extend(gap)
        output.extend(chunk)
    return output


def mix_samples(
    foreground: array,
    background: array,
    gain_foreground: float = 0.9,
    gain_background: float = 0.3,
) -> array:
    length = max(len(foreground), len(background))
    mixed = array("h", [0] * length)
    for i in range(length):
        fg = foreground[i] if i < len(foreground) else 0
        bg = background[i] if i < len(background) else 0
        value = int(fg * gain_foreground + bg * gain_background)
        if value > MAX_INT16:
            value = MAX_INT16
        elif value < -MAX_INT16:
            value = -MAX_INT16
        mixed[i] = value
    return mixed


def overlay_samples(
    base: array,
    overlay: array,
    start_index: int,
    gain: float = 1.0,
) -> array:
    if start_index >= len(base):
        return base
    end_index = min(len(base), start_index + len(overlay))
    for idx in range(start_index, end_index):
        value = base[idx] + int(overlay[idx - start_index] * gain)
        if value > MAX_INT16:
            value = MAX_INT16
        elif value < -MAX_INT16:
            value = -MAX_INT16
        base[idx] = value
    return base


def build_segment_audio(
    line_files_with_offsets: Iterable[tuple[Path, int]],
    segment_duration_ms: int,
    output_path: Path,
) -> Path:
    """Render a single WAV that overlays per-line audios at their offsets.

    Used by the video agent to build the clean dialogue track for a single
    speaker segment, which is then fed to a lip-sync model alongside the
    matching speaker portrait clip.
    """
    duration_ms = max(int(segment_duration_ms), 0)
    base = generate_silence_samples(duration_ms)
    for line_file, offset_ms in line_files_with_offsets:
        _, line_samples = read_wav(line_file)
        offset_index = int(SAMPLE_RATE * max(int(offset_ms), 0) / 1000)
        overlay_samples(base, line_samples, offset_index)
    write_wav(output_path, base)
    return output_path
