from __future__ import annotations

from pathlib import Path


class SubtitleTool:
	def build_srt(
		self,
		entries: list[tuple[int, int, str]],
		output_path: Path,
	) -> Path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as handle:
			for index, (start_ms, end_ms, text) in enumerate(entries, start=1):
				handle.write(f"{index}\n")
				handle.write(f"{self._format_time(start_ms)} --> {self._format_time(end_ms)}\n")
				handle.write(f"{text}\n\n")
		return output_path

	@staticmethod
	def _format_time(ms: int) -> str:
		hours = ms // 3_600_000
		minutes = (ms % 3_600_000) // 60_000
		seconds = (ms % 60_000) // 1000
		millis = ms % 1000
		return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
