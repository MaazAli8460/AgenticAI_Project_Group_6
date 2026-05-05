from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.tools.audio_tools.audio_utils import build_segment_audio
from mcp.tools.video_tools.compositor_tool import CompositorTool
from mcp.tools.video_tools.ffmpeg_tool import FfmpegTool
from mcp.tools.video_tools.subtitle_tool import SubtitleTool
from mcp.tools.video_tools.wav2lip_tool import Wav2LipError, Wav2LipTool
from mcp.tools.vision_tools.image_gen_tool import ImageGenTool
from shared.schemas.state import (
	PhaseStatus,
	PhaseStatusMap,
	PipelineState,
	SceneAsset,
	VideoState,
)


@dataclass
class _SceneSegment:
	image_path: Path
	duration_ms: int
	speaker_id: Optional[str] = None
	line_audio_paths: list[tuple[Path, int]] = field(default_factory=list)


class VideoAgent:
	def __init__(
		self,
		image_tool: Optional[ImageGenTool] = None,
		ffmpeg_tool: Optional[FfmpegTool] = None,
		compositor_tool: Optional[CompositorTool] = None,
		subtitle_tool: Optional[SubtitleTool] = None,
		lip_sync_tool: Optional[Wav2LipTool] = None,
		resolution: Optional[str] = None,
		fps: Optional[int] = None,
		effect: Optional[str] = None,
		subtitles: Optional[bool] = None,
		lip_sync_enabled: Optional[bool] = None,
		lip_sync_strict: Optional[bool] = None,
	) -> None:
		self._image_tool = image_tool or ImageGenTool()
		self._ffmpeg = ffmpeg_tool or FfmpegTool()
		self._compositor = compositor_tool or CompositorTool(self._ffmpeg)
		self._subtitles = subtitle_tool or SubtitleTool()
		self._resolution = resolution or os.getenv("VIDEO_RESOLUTION", "1280x720")
		self._fps = fps or int(os.getenv("VIDEO_FPS", "24"))
		self._effect = effect or os.getenv("VIDEO_EFFECT", "zoom_in")
		self._subtitles_enabled = subtitles if subtitles is not None else os.getenv(
			"SUBTITLES_ENABLED", "1"
		).lower() not in {"0", "false", "no"}
		self._seed = os.getenv("IMAGE_SEED")

		self._lip_sync = lip_sync_tool if lip_sync_tool is not None else Wav2LipTool()
		requested_enabled = (
			lip_sync_enabled
			if lip_sync_enabled is not None
			else os.getenv("LIP_SYNC_ENABLED", "0").lower() not in {"0", "false", "no"}
		)
		self._lip_sync_strict = (
			lip_sync_strict
			if lip_sync_strict is not None
			else os.getenv("LIP_SYNC_STRICT", "0").lower() in {"1", "true", "yes"}
		)
		self._lip_sync_debug = os.getenv("LIP_SYNC_DEBUG", "0").lower() in {
			"1",
			"true",
			"yes",
		}
		# Auto-disable when configuration is missing unless strict mode is set.
		availability = self._lip_sync.availability_reason()
		if requested_enabled and availability is not None and not self._lip_sync_strict:
			if self._lip_sync_debug:
				print(
					f"[lip-sync] disabling lip-sync: {availability}",
					flush=True,
				)
			requested_enabled = False
		elif requested_enabled and availability is not None and self._lip_sync_strict:
			raise RuntimeError(f"Lip-sync is enabled but Wav2Lip is unavailable: {availability}")
		self._lip_sync_enabled = requested_enabled

	def generate(self, state: PipelineState, output_dir: Path) -> PipelineState:
		if not state.scenes:
			raise ValueError("Phase 3 requires at least one scene.")
		if not state.audio or not state.audio.final_audio_file:
			raise ValueError("Phase 3 requires Phase 2 audio output.")

		width, height = self._parse_resolution(self._resolution)
		seed = None
		if self._seed:
			seed = int(self._seed)

		project_root = output_dir / state.meta.project_id
		images_dir = project_root / "images"
		clips_dir = project_root / "clips"
		final_dir = project_root / "final"
		subtitles_dir = project_root / "subtitles"
		lip_sync_dir = project_root / "lip_sync"

		scene_assets: list[SceneAsset] = []
		clips: list[Path] = []
		prompt_manifest: dict[str, dict[str, object]] = {}

		characters = {character.id: character for character in state.characters}
		scenes = sorted(state.scenes, key=lambda item: item.order or 0)
		line_map = {line.line_id: line for line in state.audio.lines}
		for scene in scenes:
			scene_entries = [
				entry
				for entry in state.audio.timing_manifest
				if entry.scene_id == scene.id
			]
			speaker_ids = self._collect_speaker_ids(scene, scene_entries, line_map)
			speaker_images, speaker_prompts = self._build_speaker_images(
				scene,
				characters,
				images_dir,
				width,
				height,
				seed,
				speaker_ids,
			)
			segments = self._build_scene_segments(
				scene,
				state,
				line_map,
				speaker_images,
				scene_entries,
			)
			segment_clips: list[Path] = []
			lip_sync_log: list[dict[str, object]] = []
			for index, segment in enumerate(segments, start=1):
				duration_s = max(segment.duration_ms / 1000.0, 0.1)
				raw_clip_path = clips_dir / f"{scene.id}_seg_{index}_raw.mp4"
				if not raw_clip_path.exists():
					self._ffmpeg.image_to_clip(
						segment.image_path,
						raw_clip_path,
						duration_s,
						width,
						height,
						self._fps,
						effect=self._effect,
					)
				final_segment_path = clips_dir / f"{scene.id}_seg_{index}.mp4"
				lip_sync_status = self._maybe_lip_sync_segment(
					segment,
					raw_clip_path,
					final_segment_path,
					lip_sync_dir,
					scene.id,
					index,
					width,
					height,
				)
				if lip_sync_status is None:
					# No lip-sync attempted (disabled or no dialogue): use raw clip directly.
					if final_segment_path != raw_clip_path:
						if final_segment_path.exists():
							final_segment_path.unlink()
						raw_clip_path.replace(final_segment_path)
				else:
					lip_sync_log.append(lip_sync_status)
				segment_clips.append(final_segment_path)

			scene_clip_path = clips_dir / f"{scene.id}.mp4"
			self._compositor.concat(segment_clips, scene_clip_path)
			clips.append(scene_clip_path)

			image_files = [str(path) for path in speaker_images.values()]
			scene_assets.append(
				SceneAsset(
					scene_id=scene.id,
					image_files=image_files,
					clip_file=str(scene_clip_path),
				)
			)
			prompt_manifest[scene.id] = {
				"speaker_prompts": speaker_prompts,
				"effect": self._effect,
				"resolution": f"{width}x{height}",
				"lip_sync_enabled": self._lip_sync_enabled,
				"lip_sync_segments": lip_sync_log,
			}

		final_dir.mkdir(parents=True, exist_ok=True)
		concat_path = final_dir / f"{state.meta.project_id}_visual.mp4"
		self._compositor.concat(clips, concat_path)

		final_audio = Path(state.audio.final_audio_file)
		final_video_path = final_dir / f"{state.meta.project_id}_final.mp4"
		self._compositor.mux_audio(concat_path, final_audio, final_video_path)

		subtitle_path: Optional[Path] = None
		if self._subtitles_enabled and state.audio.timing_manifest:
			entries = self._build_subtitle_entries(state)
			subtitle_path = subtitles_dir / f"{state.meta.project_id}.srt"
			self._subtitles.build_srt(entries, subtitle_path)
			burned_path = final_dir / f"{state.meta.project_id}_subtitled.mp4"
			self._compositor.burn_subtitles(final_video_path, subtitle_path, burned_path)
			final_video_path = burned_path

		prompt_path = project_root / "prompts.json"
		prompt_path.write_text(json.dumps(prompt_manifest, indent=2), encoding="utf-8")

		state.video = VideoState(
			scene_assets=scene_assets,
			final_video_file=str(final_video_path),
			subtitle_file=str(subtitle_path) if subtitle_path else None,
			resolution=f"{width}x{height}",
			fps=float(self._fps),
		)

		now = datetime.now(timezone.utc)
		state.meta.updated_at = now
		if state.phases is None:
			state.phases = PhaseStatusMap()
		state.phases.video = PhaseStatus(
			status="complete", updated_at=now, message="Video generated"
		)

		return state

	def _maybe_lip_sync_segment(
		self,
		segment: _SceneSegment,
		raw_clip_path: Path,
		final_segment_path: Path,
		lip_sync_dir: Path,
		scene_id: str,
		index: int,
		width: int,
		height: int,
	) -> Optional[dict[str, object]]:
		"""Run Wav2Lip on a single segment, returning a status entry or None.

		Returns ``None`` when lip-sync is skipped (disabled or no dialogue),
		signalling the caller to use the raw clip as the final segment clip.
		On Wav2Lip failure with strict=False, falls back to the raw clip and
		returns a status entry with ``"status": "fallback"``.
		"""
		if not self._lip_sync_enabled or not segment.line_audio_paths:
			return None

		lip_sync_dir.mkdir(parents=True, exist_ok=True)
		segment_audio_path = lip_sync_dir / f"{scene_id}_seg_{index}_dialogue.wav"
		if not segment_audio_path.exists():
			try:
				build_segment_audio(
					segment.line_audio_paths,
					segment.duration_ms,
					segment_audio_path,
				)
			except Exception as exc:
				if self._lip_sync_strict:
					raise
				if self._lip_sync_debug:
					print(f"[lip-sync] segment {scene_id}#{index}: failed to build audio: {exc}", flush=True)
				return {"segment": index, "status": "fallback", "reason": f"audio build failed: {exc}"}

		lip_synced_path = lip_sync_dir / f"{scene_id}_seg_{index}_synced.mp4"
		if not lip_synced_path.exists():
			try:
				self._lip_sync.lip_sync(
					raw_clip_path,
					segment_audio_path,
					lip_synced_path,
				)
			except Wav2LipError as exc:
				if self._lip_sync_strict:
					raise
				if self._lip_sync_debug:
					print(f"[lip-sync] segment {scene_id}#{index}: wav2lip failed, falling back: {exc}", flush=True)
				return {"segment": index, "status": "fallback", "reason": str(exc).splitlines()[0] if str(exc) else "wav2lip error"}

		try:
			self._ffmpeg.normalize_clip(
				lip_synced_path,
				final_segment_path,
				width,
				height,
				self._fps,
			)
		except Exception as exc:
			if self._lip_sync_strict:
				raise
			if self._lip_sync_debug:
				print(
					f"[lip-sync] segment {scene_id}#{index}: normalize failed, falling back: {exc}",
					flush=True,
				)
			return {
				"segment": index,
				"status": "fallback",
				"reason": f"normalize failed: {exc}",
			}

		if self._lip_sync_debug:
			print(
				f"[lip-sync] segment {scene_id}#{index}: lip-synced "
				f"({segment.speaker_id or 'unknown'})",
				flush=True,
			)
		return {
			"segment": index,
			"status": "lip_synced",
			"speaker_id": segment.speaker_id,
			"audio_file": str(segment_audio_path),
			"raw_clip": str(raw_clip_path),
			"synced_clip": str(lip_synced_path),
		}

	@staticmethod
	def _parse_resolution(value: str) -> tuple[int, int]:
		parts = value.lower().split("x")
		if len(parts) != 2:
			raise ValueError("VIDEO_RESOLUTION must be WIDTHxHEIGHT")
		return int(parts[0]), int(parts[1])

	@staticmethod
	def _collect_speaker_ids(scene, scene_entries: list, line_map: dict) -> list[str]:
		speaker_ids: list[str] = []
		for entry in scene_entries:
			line = line_map.get(entry.line_id)
			speaker_id = line.character_id if line else entry.character_id
			if speaker_id:
				speaker_ids.append(speaker_id)
		if not speaker_ids:
			speaker_ids = list(scene.character_ids)
		return list(dict.fromkeys(speaker_ids))

	@staticmethod
	def _voice_gender_hint(character) -> Optional[str]:
		gender = getattr(character.voice, "gender", None)
		if not gender:
			return None
		slug = str(gender).strip().lower()
		if slug in {"male", "man", "boy", "masculine", "m"}:
			return "male"
		if slug in {"female", "woman", "girl", "feminine", "f"}:
			return "female"
		if slug in {"neutral", "nonbinary", "non-binary", "androgynous"}:
			return "androgynous"
		return None

	@staticmethod
	def _build_speaker_prompt(scene, character) -> str:
		gender_hint = VideoAgent._voice_gender_hint(character)
		if gender_hint:
			speaking_prompt = (
				"photorealistic, sharp focus, high detail, no blur, no distortion, "
				"camera-facing, front-facing close-up portrait of a "
				f"{gender_hint} character named {character.name} speaking"
			)
		else:
			speaking_prompt = (
				"photorealistic, sharp focus, high detail, no blur, no distortion, "
				"camera-facing, front-facing close-up portrait of "
				f"{character.name} speaking"
			)
		parts = [
			scene.visual_prompt,
			scene.mood,
			scene.style,
			speaking_prompt,
			character.visual.description,
		]
		return ", ".join(part for part in parts if part)

	def _build_speaker_images(
		self,
		scene,
		characters: dict,
		images_dir: Path,
		width: int,
		height: int,
		seed: Optional[int],
		speaker_ids: Optional[list[str]] = None,
	) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
		speaker_ids = speaker_ids or []
		if not speaker_ids:
			speaker_ids = list(dict.fromkeys(scene.character_ids))
		speaker_images: dict[str, Path] = {}
		speaker_prompts: dict[str, dict[str, str]] = {}
		for speaker_id in speaker_ids:
			character = characters.get(speaker_id)
			if not character:
				continue
			prompt = self._build_speaker_prompt(scene, character)
			image_path = images_dir / f"{scene.id}_{speaker_id}.png"
			if not image_path.exists():
				self._image_tool.generate(prompt, image_path, width, height, seed=seed)
			self._validate_image_file(image_path, scene.id)
			speaker_images[speaker_id] = image_path
			speaker_prompts[speaker_id] = {
				"prompt": prompt,
				"image": str(image_path),
			}
		if not speaker_images:
			fallback_prompt = scene.visual_prompt
			fallback_path = images_dir / f"{scene.id}_scene.png"
			if not fallback_path.exists():
				self._image_tool.generate(fallback_prompt, fallback_path, width, height, seed=seed)
			self._validate_image_file(fallback_path, scene.id)
			speaker_images["scene"] = fallback_path
			speaker_prompts["scene"] = {
				"prompt": fallback_prompt,
				"image": str(fallback_path),
			}
		return speaker_images, speaker_prompts

	def _build_scene_segments(
		self,
		scene,
		state: PipelineState,
		line_map: dict,
		speaker_images: dict[str, Path],
		scene_entries: list,
	) -> list[_SceneSegment]:
		entries = sorted(scene_entries, key=lambda entry: entry.start_ms)
		fallback_image = next(iter(speaker_images.values()))

		if not entries:
			fallback_duration = int(self._scene_duration_s(scene.id, state) * 1000)
			return [
				_SceneSegment(
					image_path=fallback_image,
					duration_ms=fallback_duration,
				)
			]

		segments: list[_SceneSegment] = []
		current: Optional[_SceneSegment] = None
		current_start: Optional[int] = None
		current_end: Optional[int] = None

		def _close_segment(end_ms: Optional[int]) -> None:
			nonlocal current, current_start, current_end
			if current is None or current_start is None:
				return
			final_end = end_ms if end_ms is not None else current_end
			if final_end is None:
				final_end = current_start
			current.duration_ms = max(int(final_end) - int(current_start), 100)
			segments.append(current)
			current = None
			current_start = None
			current_end = None

		for entry in entries:
			line = line_map.get(entry.line_id)
			speaker_id = line.character_id if line else entry.character_id
			if not speaker_id:
				speaker_id = (
					current.speaker_id
					if current is not None
					else next(iter(speaker_images.keys()), None)
				)
			image_path = speaker_images.get(speaker_id, fallback_image) if speaker_id else fallback_image
			audio_file = Path(entry.audio_file) if entry.audio_file else None

			if current is None:
				current = _SceneSegment(
					image_path=image_path,
					duration_ms=0,
					speaker_id=speaker_id,
				)
				current_start = entry.start_ms
				current_end = entry.end_ms
				if audio_file is not None:
					current.line_audio_paths.append(
						(audio_file, entry.start_ms - (current_start or 0))
					)
				continue

			if speaker_id == current.speaker_id:
				current_end = max(current_end or entry.end_ms, entry.end_ms)
				if audio_file is not None and current_start is not None:
					current.line_audio_paths.append(
						(audio_file, entry.start_ms - current_start)
					)
				continue

			# Speaker change: close out the running segment and start a new one.
			segment_end = max(current_end or entry.start_ms, entry.start_ms)
			_close_segment(segment_end)
			current = _SceneSegment(
				image_path=image_path,
				duration_ms=0,
				speaker_id=speaker_id,
			)
			current_start = entry.start_ms
			current_end = entry.end_ms
			if audio_file is not None:
				current.line_audio_paths.append((audio_file, 0))

		_close_segment(current_end)

		target_ms = int(self._scene_duration_s(scene.id, state) * 1000)
		current_ms = sum(seg.duration_ms for seg in segments)
		if current_ms < target_ms and segments:
			segments[-1].duration_ms += target_ms - current_ms
		elif current_ms > target_ms and segments:
			excess = current_ms - target_ms
			segments[-1].duration_ms = max(segments[-1].duration_ms - excess, 100)
		return segments

	@staticmethod
	def _validate_image_file(path: Path, scene_id: str) -> None:
		if not path.exists() or path.stat().st_size == 0:
			raise RuntimeError(f"Image generation failed for {scene_id}: file missing.")
		with path.open("rb") as handle:
			header = handle.read(12)
		if header.startswith(b"\x89PNG\r\n\x1a\n"):
			return
		if header.startswith(b"\xff\xd8"):
			return
		if header.startswith(b"RIFF") and b"WEBP" in header:
			return
		raise RuntimeError(
			f"Image generation failed for {scene_id}: non-image content received."
		)

	def _scene_duration_s(self, scene_id: str, state: PipelineState) -> float:
		if state.audio and state.audio.bgm_tracks:
			for track in state.audio.bgm_tracks:
				if track.scene_id == scene_id:
					return max((track.end_ms - track.start_ms) / 1000.0, 0.1)
		if state.audio and state.audio.timing_manifest:
			start = None
			end = None
			for entry in state.audio.timing_manifest:
				if entry.scene_id != scene_id:
					continue
				start = entry.start_ms if start is None else min(start, entry.start_ms)
				end = entry.end_ms if end is None else max(end, entry.end_ms)
			if start is not None and end is not None:
				return max((end - start) / 1000.0, 0.1)
		for scene in state.scenes:
			if scene.id == scene_id:
				return max(scene.duration_s, 0.1)
		return 1.0

	@staticmethod
	def _build_subtitle_entries(state: PipelineState) -> list[tuple[int, int, str]]:
		text_map = {line.line_id: line.text for line in state.audio.lines}
		entries = []
		for entry in state.audio.timing_manifest:
			text = text_map.get(entry.line_id)
			if not text:
				continue
			entries.append((entry.start_ms, entry.end_ms, text))
		return entries
