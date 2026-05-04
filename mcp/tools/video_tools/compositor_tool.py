from __future__ import annotations

from pathlib import Path

from .ffmpeg_tool import FfmpegTool


class CompositorTool:
	def __init__(self, ffmpeg_tool: FfmpegTool) -> None:
		self._ffmpeg = ffmpeg_tool

	def concat(self, clips: list[Path], output_path: Path) -> Path:
		list_path = output_path.parent / "concat.txt"
		list_path.parent.mkdir(parents=True, exist_ok=True)
		with list_path.open("w", encoding="utf-8") as handle:
			for clip in clips:
				clip_path = clip.resolve().as_posix()
				handle.write(f"file '{clip_path}'\n")
		return self._ffmpeg.concat_clips(list_path, output_path)

	def mux_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
		return self._ffmpeg.mux_audio(video_path, audio_path, output_path)

	def burn_subtitles(self, video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
		return self._ffmpeg.add_subtitles(video_path, subtitle_path, output_path)
