from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from shared.constants.bgm_moods import BGM_DEFAULT_MOOD, normalize_bgm_mood

from .audio_utils import generate_tone_samples, write_wav


MOOD_FREQUENCIES = {
	"ambient": 200.0,
	"mystical": 180.0,
	"reflective": 210.0,
	"curious": 230.0,
	"empowering": 260.0,
	"hopeful": 240.0,
}


class BgmTool:
	def __init__(self, library_dir: Optional[Path] = None) -> None:
		env_dir = os.getenv("BGM_LIBRARY_DIR")
		self._library_dir = Path(env_dir) if env_dir else library_dir
		self._source_cycle: dict[str, list[Path]] = {}
		self._source_index: dict[str, int] = {}
		self._ffmpeg = shutil.which("ffmpeg")
		self._debug = os.getenv("BGM_DEBUG", "").lower() in {"1", "true", "yes"}
		self._strict = os.getenv("BGM_STRICT", "").lower() in {"1", "true", "yes"}
		self._start_offset_s = float(os.getenv("BGM_START_OFFSET_S", "20"))
		seed_value = os.getenv("BGM_RANDOM_SEED")
		self._rng = random.Random(int(seed_value)) if seed_value else random.Random()

	def generate(
		self,
		scene_id: str,
		mood: Optional[str],
		duration_ms: int,
		output_path: Path,
	) -> Path:
		mood_key = normalize_bgm_mood(mood)
		source = self._pick_library_track(mood_key)
		if source:
			output_path.parent.mkdir(parents=True, exist_ok=True)
			if self._ffmpeg:
				if self._convert_to_wav(source, output_path, duration_ms):
					self._maybe_warn(f"BGM {scene_id} using {source}")
					return output_path
			if source.suffix.lower() == ".wav":
				shutil.copyfile(source, output_path)
				self._maybe_warn(f"BGM {scene_id} using {source}")
				return output_path
			self._maybe_warn("FFmpeg missing or conversion failed; using fallback tone.")
			if self._strict:
				raise RuntimeError("BGM conversion failed; install FFmpeg or provide WAVs.")

		frequency = MOOD_FREQUENCIES.get(mood_key, 200.0)
		samples = generate_tone_samples(duration_ms, frequency_hz=frequency, volume=0.06)
		write_wav(output_path, samples)
		return output_path

	def _pick_library_track(self, mood: str) -> Optional[Path]:
		if not self._library_dir or not self._library_dir.exists():
			return None
		mood_dir = self._library_dir / mood
		if not mood_dir.exists() or not mood_dir.is_dir():
			mood_dir = self._library_dir / BGM_DEFAULT_MOOD
		if not mood_dir.exists() or not mood_dir.is_dir():
			return None
		candidates = sorted(
			[
				*list(mood_dir.glob("*.wav")),
				*list(mood_dir.glob("*.mp3")),
				*list(mood_dir.glob("*.m4a")),
				*list(mood_dir.glob("*.flac")),
				*list(mood_dir.glob("*.ogg")),
			]
		)
		if not candidates:
			return None
		cycle = self._source_cycle.get(mood)
		index = self._source_index.get(mood, 0)
		if not cycle:
			cycle = candidates
			self._rng.shuffle(cycle)
			index = 0
		if index >= len(cycle):
			self._rng.shuffle(cycle)
			index = 0
		track = cycle[index]
		self._source_cycle[mood] = cycle
		self._source_index[mood] = index + 1
		return track

	def _convert_to_wav(self, source: Path, output_path: Path, duration_ms: int) -> bool:
		if not self._ffmpeg:
			return False
		try:
			duration_s = max(duration_ms / 1000.0, 0.1)
			start_offset = max(self._start_offset_s, 0.0)
			result = subprocess.run(
				[
					self._ffmpeg,
					"-y",
					"-ss",
					f"{start_offset}",
					"-i",
					str(source),
					"-t",
					f"{duration_s}",
					"-ac",
					"1",
					"-ar",
					"22050",
					"-f",
					"wav",
					str(output_path),
				],
				capture_output=True,
				check=True,
			)
		except (subprocess.CalledProcessError, FileNotFoundError):
			return False
		return output_path.exists()

	def _maybe_warn(self, message: str) -> None:
		if self._debug:
			print(f"[BGM] {message}", file=sys.stderr)
