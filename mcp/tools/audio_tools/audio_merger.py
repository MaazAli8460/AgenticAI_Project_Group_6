from __future__ import annotations

from array import array
from pathlib import Path
from typing import Iterable, Optional

from .audio_utils import (
	SAMPLE_RATE,
	concat_samples,
	mix_samples,
	pad_samples_to_duration,
	read_wav,
	write_wav,
)


class AudioMerger:
	def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
		self._sample_rate = sample_rate

	def concatenate(
		self,
		segments: Iterable[Path],
		output_path: Path,
		gap_ms: int = 200,
		target_duration_ms: Optional[int] = None,
	) -> Path:
		chunks: list[array] = []
		for segment in segments:
			sample_rate, samples = read_wav(segment)
			self._ensure_sample_rate(sample_rate)
			chunks.append(samples)
		combined = concat_samples(chunks, gap_ms=gap_ms)
		if target_duration_ms is not None:
			combined = pad_samples_to_duration(combined, target_duration_ms)
		write_wav(output_path, combined)
		return output_path

	def mix(
		self,
		dialogue_path: Path,
		bgm_path: Path,
		output_path: Path,
		gain_dialogue: float = 0.9,
		gain_bgm: float = 0.3,
		target_duration_ms: Optional[int] = None,
	) -> Path:
		sample_rate_dialogue, dialogue = read_wav(dialogue_path)
		sample_rate_bgm, bgm = read_wav(bgm_path)
		self._ensure_sample_rate(sample_rate_dialogue)
		self._ensure_sample_rate(sample_rate_bgm)
		mixed = mix_samples(dialogue, bgm, gain_dialogue, gain_bgm)
		if target_duration_ms is not None:
			mixed = pad_samples_to_duration(mixed, target_duration_ms)
		write_wav(output_path, mixed)
		return output_path

	def _ensure_sample_rate(self, sample_rate: int) -> None:
		if sample_rate != self._sample_rate:
			raise ValueError("Audio sample rate mismatch during merge.")
