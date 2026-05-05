from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _find_ffmpeg_via_winget_windows() -> Path | None:
	"""Locate ffmpeg.exe installed by WinGet (PATH is often stale until shell restart)."""
	local = os.environ.get("LOCALAPPDATA")
	if not local:
		return None
	packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
	if not packages.is_dir():
		return None
	# e.g. .../Gyan.FFmpeg_.../ffmpeg-8.1.1-full_build/bin/ffmpeg.exe
	candidates: list[Path] = []
	for pkg_root in packages.iterdir():
		if "ffmpeg" not in pkg_root.name.lower():
			continue
		for exe in pkg_root.rglob("ffmpeg.exe"):
			if "bin" in exe.parts:
				candidates.append(exe)
	if not candidates:
		return None
	return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_ffmpeg_executable(explicit: str | Path | None = None) -> str | None:
	"""Return an ffmpeg executable path, or None if none found.

	Order: explicit arg, ``FFMPEG_PATH`` env, ``FFMPEG`` env, ``PATH`` (``which``),
	then WinGet default layout on Windows.
	"""
	if explicit:
		p = Path(explicit)
		if p.is_file():
			return str(p.resolve())
	if env := (os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG") or "").strip():
		p = Path(env)
		if p.is_file():
			return str(p.resolve())
	found = shutil.which("ffmpeg")
	if found:
		return found
	wg = _find_ffmpeg_via_winget_windows()
	return str(wg.resolve()) if wg and wg.is_file() else None


class FfmpegTool:
	def __init__(self, ffmpeg_path: str | None = None) -> None:
		self._ffmpeg = resolve_ffmpeg_executable(ffmpeg_path)
		if not self._ffmpeg:
			raise RuntimeError(
				"FFmpeg is required for Phase 3 video rendering. Install it and ensure "
				"it is on PATH, or set FFMPEG_PATH to the full path of ffmpeg.exe "
				'(e.g. after `winget install Gyan.FFmpeg`, restart the terminal or set '
				"FFMPEG_PATH to ...\\WinGet\\Packages\\...\\bin\\ffmpeg.exe)."
			)

	def image_to_clip(
		self,
		image_path: Path,
		output_path: Path,
		duration_s: float,
		width: int,
		height: int,
		fps: int,
		effect: str = "zoom_in",
	) -> Path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		frames = max(int(duration_s * fps), 1)
		filter_chain = self._build_filter(effect, width, height, frames)
		cmd = [
			self._ffmpeg,
			"-y",
			"-loop",
			"1",
			"-i",
			str(image_path),
			"-t",
			f"{duration_s}",
			"-vf",
			filter_chain,
			"-r",
			str(fps),
			"-pix_fmt",
			"yuv420p",
			"-c:v",
			"libx264",
			str(output_path),
		]
		self._run(cmd)
		return output_path

	def concat_clips(self, list_file: Path, output_path: Path) -> Path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		cmd = [
			self._ffmpeg,
			"-y",
			"-f",
			"concat",
			"-safe",
			"0",
			"-i",
			str(list_file),
			"-c:v",
			"libx264",
			"-pix_fmt",
			"yuv420p",
			str(output_path),
		]
		self._run(cmd)
		return output_path

	def mux_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		if not video_path.exists():
			raise RuntimeError(f"Video file missing: {video_path}")
		if not audio_path.exists():
			raise RuntimeError(f"Audio file missing: {audio_path}")
		cmd = [
			self._ffmpeg,
			"-y",
			"-i",
			str(video_path),
			"-i",
			str(audio_path),
			"-map",
			"0:v:0",
			"-map",
			"1:a:0",
			"-c:v",
			"copy",
			"-c:a",
			"aac",
			"-shortest",
			str(output_path),
		]
		self._run(cmd)
		return output_path

	def normalize_clip(
		self,
		input_path: Path,
		output_path: Path,
		width: int,
		height: int,
		fps: int,
	) -> Path:
		"""Re-encode a clip to a uniform spec (h264, yuv420p, scale, fps, no audio).

		Used after lip-sync passes (which may emit non-h264 or differently sized
		MP4s) so all segment clips are mutually concat-compatible.
		"""
		output_path.parent.mkdir(parents=True, exist_ok=True)
		if not input_path.exists():
			raise RuntimeError(f"Input clip missing: {input_path}")
		cmd = [
			self._ffmpeg,
			"-y",
			"-i",
			str(input_path),
			"-vf",
			f"scale={width}:{height},fps={fps}",
			"-pix_fmt",
			"yuv420p",
			"-c:v",
			"libx264",
			"-an",
			str(output_path),
		]
		self._run(cmd)
		return output_path

	def add_subtitles(self, video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		if not video_path.exists():
			raise RuntimeError(f"Video file missing: {video_path}")
		if not subtitle_path.exists():
			raise RuntimeError(f"Subtitle file missing: {subtitle_path}")
		subtitle_filter = self._subtitle_filter_path(subtitle_path)
		cmd = [
			self._ffmpeg,
			"-y",
			"-i",
			str(video_path),
			"-vf",
			subtitle_filter,
			"-map",
			"0:v:0",
			"-map",
			"0:a?",
			"-c:v",
			"libx264",
			"-c:a",
			"copy",
			str(output_path),
		]
		self._run(cmd)
		return output_path

	@staticmethod
	def _build_filter(effect: str, width: int, height: int, frames: int) -> str:
		scale = f"scale={width}:{height}"
		if effect == "none":
			return scale
		if effect == "zoom_out":
			zoom = "z='max(1.0,1.15-0.001*on)'"
		else:
			zoom = "z='min(1.15,1.0+0.001*on)'"
		return f"{scale},zoompan={zoom}:d={frames}:s={width}x{height}"

	@staticmethod
	def _run(cmd: list[str]) -> None:
		try:
			result = subprocess.run(cmd, check=True, capture_output=True)
		except subprocess.CalledProcessError as exc:
			stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
			stdout = exc.stdout.decode(errors="ignore") if exc.stdout else ""
			raise RuntimeError(
				"FFmpeg command failed.\n"
				f"Command: {' '.join(cmd)}\n"
				f"Stdout: {stdout}\n"
				f"Stderr: {stderr}"
			) from exc

	@staticmethod
	def _subtitle_filter_path(path: Path) -> str:
		posix_path = path.as_posix()
		escaped = posix_path.replace("\\", "/").replace(":", "\\:").replace(" ", "\\ ")
		return f"subtitles={escaped}"
