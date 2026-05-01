from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

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

	def generate(
		self,
		scene_id: str,
		mood: Optional[str],
		duration_ms: int,
		output_path: Path,
	) -> Path:
		source = self._pick_library_track(mood)
		if source:
			output_path.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(source, output_path)
			return output_path

		frequency = MOOD_FREQUENCIES.get((mood or "").lower(), 200.0)
		samples = generate_tone_samples(duration_ms, frequency_hz=frequency, volume=0.1)
		write_wav(output_path, samples)
		return output_path

	def _pick_library_track(self, mood: Optional[str]) -> Optional[Path]:
		if not self._library_dir or not self._library_dir.exists():
			return None
		candidates = sorted(self._library_dir.glob("*.wav"))
		if not candidates:
			return None
		if mood:
			mood_key = mood.lower()
			matches = [track for track in candidates if mood_key in track.stem.lower()]
			if matches:
				return matches[0]
		return candidates[0]
