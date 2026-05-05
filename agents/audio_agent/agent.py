from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Optional

from mcp.tools.audio_tools.audio_merger import AudioMerger
from mcp.tools.audio_tools.audio_utils import (
	SAMPLE_RATE,
	generate_silence_samples,
	overlay_samples,
	pad_samples_to_duration,
	read_wav,
	write_wav,
)
from mcp.tools.audio_tools.bgm_tool import BgmTool
from shared.constants.bgm_moods import normalize_bgm_mood
from mcp.tools.audio_tools.tts_tool import TtsTool
from shared.schemas.state import (
	AudioLine,
	AudioState,
	BgmTrack,
	PhaseStatus,
	PhaseStatusMap,
	PipelineState,
	TimingEntry,
)


class AudioAgent:
	def __init__(
		self,
		tts_tool: Optional[TtsTool] = None,
		bgm_tool: Optional[BgmTool] = None,
		audio_merger: Optional[AudioMerger] = None,
		gap_ms: int = 200,
	) -> None:
		self._tts = tts_tool or TtsTool()
		self._bgm = bgm_tool or BgmTool()
		self._merger = audio_merger or AudioMerger()
		self._gap_ms = gap_ms
		self._bgm_gain = float(os.getenv("BGM_GAIN", "0.2"))
		self._dialogue_gain = float(os.getenv("DIALOGUE_GAIN", "0.9"))

	def generate(self, state: PipelineState, output_dir: Path) -> PipelineState:
		if not state.scenes:
			raise ValueError("Phase 2 requires at least one scene.")
		if not state.characters:
			raise ValueError("Phase 2 requires at least one character.")

		project_root = output_dir / state.meta.project_id
		lines_dir = project_root / "lines"
		bgm_dir = project_root / "bgm"
		scenes_dir = project_root / "scenes"
		final_dir = project_root / "final"

		character_map = {character.id: character for character in state.characters}

		audio_lines: list[AudioLine] = []
		timing_manifest: list[TimingEntry] = []
		bgm_tracks: list[BgmTrack] = []
		scene_mix_paths: list[Path] = []

		timeline_ms = 0
		scenes = sorted(state.scenes, key=lambda item: item.order or 0)
		for scene in scenes:
			line_items: list[tuple[Path, int]] = []
			scene_start_ms = timeline_ms
			scene_local_ms = 0
			last_line_end_local = 0

			for line_index, line in enumerate(scene.dialogue, start=1):
				character = character_map.get(line.character_id, state.characters[0])
				file_name = line.id or f"line_{scene.id}_{line_index}"
				line_path = lines_dir / f"{file_name}.wav"
				duration_ms = self._tts.synthesize(line.text, character.voice, line_path)
				start_ms = scene_start_ms + scene_local_ms
				end_ms = start_ms + duration_ms

				timing_manifest.append(
					TimingEntry(
						scene_id=scene.id,
						line_id=line.id or file_name,
						character_id=character.id,
						audio_file=str(line_path),
						start_ms=start_ms,
						end_ms=end_ms,
					)
				)
				audio_lines.append(
					AudioLine(
						line_id=line.id or file_name,
						scene_id=scene.id,
						character_id=character.id,
						audio_file=str(line_path),
						text=line.text,
						start_ms=start_ms,
						end_ms=end_ms,
						voice_params=character.voice.params,
					)
				)

				last_line_end_local = scene_local_ms + duration_ms
				scene_local_ms = scene_local_ms + duration_ms + self._gap_ms
				line_items.append((line_path, start_ms - scene_start_ms))

			if line_items:
				scene_local_ms = max(scene_local_ms - self._gap_ms, 0)

			scene_duration_ms = max(
				int(scene.duration_s * 1000),
				int(last_line_end_local),
			)
			if scene_duration_ms <= 0:
				scene_duration_ms = 1000

			dialogue_path = scenes_dir / f"{scene.id}_dialogue.wav"
			dialogue_samples = generate_silence_samples(scene_duration_ms)
			for line_path, offset_ms in line_items:
				_, line_samples = read_wav(line_path)
				offset_index = int(SAMPLE_RATE * offset_ms / 1000)
				overlay_samples(dialogue_samples, line_samples, offset_index)
			write_wav(dialogue_path, dialogue_samples)

			bgm_path = bgm_dir / f"{scene.id}_bgm.wav"
			mood_key = normalize_bgm_mood(scene.bgm_mood or scene.mood)
			self._bgm.generate(
				scene.id,
				mood_key,
				scene_duration_ms,
				bgm_path,
			)

			mix_path = scenes_dir / f"{scene.id}_mix.wav"
			self._merger.mix(
				dialogue_path,
				bgm_path,
				mix_path,
				gain_dialogue=self._dialogue_gain,
				gain_bgm=self._bgm_gain,
				target_duration_ms=scene_duration_ms,
			)
			scene_mix_paths.append(mix_path)

			scene_end_ms = scene_start_ms + scene_duration_ms
			bgm_tracks.append(
				BgmTrack(
					scene_id=scene.id,
					audio_file=str(bgm_path),
					start_ms=scene_start_ms,
					end_ms=scene_end_ms,
					mood=mood_key,
					style=scene.bgm_style,
				)
			)
			timeline_ms = scene_end_ms

		final_audio_path = final_dir / f"{state.meta.project_id}_final.wav"
		self._merger.concatenate(scene_mix_paths, final_audio_path, gap_ms=0)

		state.audio = AudioState(
			timing_manifest=timing_manifest,
			lines=audio_lines,
			bgm_tracks=bgm_tracks,
			final_audio_file=str(final_audio_path),
		)

		now = datetime.now(timezone.utc)
		state.meta.updated_at = now
		if state.phases is None:
			state.phases = PhaseStatusMap()
		state.phases.audio = PhaseStatus(
			status="complete", updated_at=now, message="Audio generated"
		)

		return state

	def regenerate_scene_bgm(
		self,
		state: PipelineState,
		output_dir: Path,
		scene_id: str,
	) -> PipelineState:
		if not state.audio or not state.audio.timing_manifest:
			raise ValueError("Phase 2 state is required to update BGM.")

		project_root = output_dir / state.meta.project_id
		lines_dir = project_root / "lines"
		bgm_dir = project_root / "bgm"
		scenes_dir = project_root / "scenes"
		final_dir = project_root / "final"

		scene = next((item for item in state.scenes if item.id == scene_id), None)
		if scene is None:
			raise ValueError(f"Scene '{scene_id}' not found.")

		entries = sorted(
			[entry for entry in state.audio.timing_manifest if entry.scene_id == scene_id],
			key=lambda item: item.start_ms,
		)
		scene_start_ms = min((entry.start_ms for entry in entries), default=0)
		scene_end_ms = max((entry.end_ms for entry in entries), default=0)
		scene_duration_ms = max(
			int(scene.duration_s * 1000),
			int(scene_end_ms - scene_start_ms),
			1000,
		)

		dialogue_path = scenes_dir / f"{scene.id}_dialogue.wav"
		if not dialogue_path.exists():
			dialogue_samples = generate_silence_samples(scene_duration_ms)
			for entry in entries:
				line_path = Path(entry.audio_file)
				if not line_path.exists():
					fallback = lines_dir / f"{entry.line_id}.wav"
					if fallback.exists():
						line_path = fallback
					else:
						raise FileNotFoundError(f"Missing dialogue audio: {entry.audio_file}")
				_, line_samples = read_wav(line_path)
				offset_ms = entry.start_ms - scene_start_ms
				offset_index = int(SAMPLE_RATE * offset_ms / 1000)
				overlay_samples(dialogue_samples, line_samples, offset_index)
			write_wav(dialogue_path, dialogue_samples)

		bgm_path = bgm_dir / f"{scene.id}_bgm.wav"
		mood_key = normalize_bgm_mood(scene.bgm_mood or scene.mood)
		self._bgm.generate(
			scene.id,
			mood_key,
			scene_duration_ms,
			bgm_path,
		)

		mix_path = scenes_dir / f"{scene.id}_mix.wav"
		self._merger.mix(
			dialogue_path,
			bgm_path,
			mix_path,
			gain_dialogue=self._dialogue_gain,
			gain_bgm=self._bgm_gain,
			target_duration_ms=scene_duration_ms,
		)

		bgm_tracks = state.audio.bgm_tracks or []
		updated = False
		for track in bgm_tracks:
			if track.scene_id == scene.id:
				track.audio_file = str(bgm_path)
				track.start_ms = scene_start_ms
				track.end_ms = scene_start_ms + scene_duration_ms
				track.mood = mood_key
				track.style = scene.bgm_style
				updated = True
				break
			if not updated:
				bgm_tracks.append(
					BgmTrack(
						scene_id=scene.id,
						audio_file=str(bgm_path),
						start_ms=scene_start_ms,
						end_ms=scene_start_ms + scene_duration_ms,
						mood=mood_key,
						style=scene.bgm_style,
					)
				)
		state.audio.bgm_tracks = bgm_tracks

		final_dir.mkdir(parents=True, exist_ok=True)
		state.audio.final_audio_file = str(
			final_dir / f"{state.meta.project_id}_final.wav"
		)
		return state

	def regenerate_scene_dialogue(
		self,
		state: PipelineState,
		output_dir: Path,
		scene_id: str,
		preserve_timing: bool = True,
	) -> PipelineState:
		if not state.audio or not state.audio.timing_manifest:
			raise ValueError("Phase 2 state is required to update dialogue.")
		if not state.characters:
			raise ValueError("Phase 2 requires at least one character.")

		scene = next((item for item in state.scenes if item.id == scene_id), None)
		if scene is None:
			raise ValueError(f"Scene '{scene_id}' not found.")
		if not scene.dialogue:
			raise ValueError(f"Scene '{scene_id}' has no dialogue to regenerate.")

		project_root = output_dir / state.meta.project_id
		lines_dir = project_root / "lines"
		bgm_dir = project_root / "bgm"
		scenes_dir = project_root / "scenes"
		final_dir = project_root / "final"

		existing_entries = [
			entry for entry in state.audio.timing_manifest if entry.scene_id == scene_id
		]
		if not existing_entries:
			raise ValueError("Scene timing entries are missing; re-run Phase 2.")
		entries_by_line = {entry.line_id: entry for entry in existing_entries}

		missing = []
		for index, line in enumerate(scene.dialogue, start=1):
			line_id = line.id or f"line_{scene.id}_{index}"
			if preserve_timing and line_id not in entries_by_line:
				missing.append(line_id)
		if preserve_timing and missing:
			raise ValueError(
				"Missing timing entries for lines: " + ", ".join(missing)
			)

		scene_start_ms = min(entry.start_ms for entry in existing_entries)
		scene_end_ms = max(entry.end_ms for entry in existing_entries)
		scene_duration_ms = max(scene_end_ms - scene_start_ms, 1000)

		character_map = {character.id: character for character in state.characters}
		new_entries: list[TimingEntry] = []
		new_lines: list[AudioLine] = []
		line_items: list[tuple[Path, int]] = []
		scene_local_ms = 0
		last_line_end_local = 0

		for index, line in enumerate(scene.dialogue, start=1):
			character = character_map.get(line.character_id, state.characters[0])
			line_id = line.id or f"line_{scene.id}_{index}"
			line_path = lines_dir / f"{line_id}.wav"
			duration_ms = self._tts.synthesize(line.text, character.voice, line_path)

			if preserve_timing:
				entry = entries_by_line[line_id]
				start_ms = entry.start_ms
				end_ms = entry.end_ms
				target_ms = max(end_ms - start_ms, 0)
				if target_ms > 0:
					_, samples = read_wav(line_path)
					write_wav(line_path, pad_samples_to_duration(samples, target_ms))
			else:
				start_ms = scene_start_ms + scene_local_ms
				end_ms = start_ms + duration_ms
				scene_local_ms = scene_local_ms + duration_ms + self._gap_ms
				last_line_end_local = max(last_line_end_local, end_ms - scene_start_ms)

			new_entries.append(
				TimingEntry(
					scene_id=scene.id,
					line_id=line_id,
					character_id=character.id,
					audio_file=str(line_path),
					start_ms=start_ms,
					end_ms=end_ms,
				)
			)
			new_lines.append(
				AudioLine(
					line_id=line_id,
					scene_id=scene.id,
					character_id=character.id,
					audio_file=str(line_path),
					text=line.text,
					start_ms=start_ms,
					end_ms=end_ms,
					voice_params=character.voice.params,
				)
			)
			line_items.append((line_path, start_ms - scene_start_ms))

		if not preserve_timing:
			scene_duration_ms = max(scene_duration_ms, last_line_end_local, 1000)

		dialogue_path = scenes_dir / f"{scene.id}_dialogue.wav"
		dialogue_samples = generate_silence_samples(scene_duration_ms)
		for line_path, offset_ms in line_items:
			_, line_samples = read_wav(line_path)
			offset_index = int(SAMPLE_RATE * max(int(offset_ms), 0) / 1000)
			overlay_samples(dialogue_samples, line_samples, offset_index)
		write_wav(dialogue_path, dialogue_samples)

		bgm_path: Optional[Path] = None
		if state.audio.bgm_tracks:
			for track in state.audio.bgm_tracks:
				if track.scene_id == scene.id:
					bgm_path = Path(track.audio_file)
					track.start_ms = scene_start_ms
					track.end_ms = scene_start_ms + scene_duration_ms
					break
			if bgm_path is None:
				bgm_path = bgm_dir / f"{scene.id}_bgm.wav"
		else:
			bgm_path = bgm_dir / f"{scene.id}_bgm.wav"
		if not bgm_path.exists():
			raise FileNotFoundError(f"Missing BGM track for scene {scene.id}: {bgm_path}")

		mix_path = scenes_dir / f"{scene.id}_mix.wav"
		self._merger.mix(
			dialogue_path,
			bgm_path,
			mix_path,
			gain_dialogue=self._dialogue_gain,
			gain_bgm=self._bgm_gain,
			target_duration_ms=scene_duration_ms,
		)

		other_entries = [
			entry for entry in state.audio.timing_manifest if entry.scene_id != scene_id
		]
		state.audio.timing_manifest = sorted(
			other_entries + new_entries,
			key=lambda item: item.start_ms,
		)

		other_lines = [
			line for line in state.audio.lines if line.scene_id != scene_id
		]
		state.audio.lines = sorted(
			other_lines + new_lines,
			key=lambda item: item.start_ms or 0,
		)

		final_dir.mkdir(parents=True, exist_ok=True)
		state.audio.final_audio_file = str(
			final_dir / f"{state.meta.project_id}_final.wav"
		)
		return state

	def rebuild_final_audio(self, state: PipelineState, output_dir: Path) -> Path:
		if not state.audio:
			raise ValueError("Phase 2 state is required to rebuild final audio.")

		project_root = output_dir / state.meta.project_id
		scenes_dir = project_root / "scenes"
		final_dir = project_root / "final"
		final_dir.mkdir(parents=True, exist_ok=True)

		mix_paths: list[Path] = []
		for scene in sorted(state.scenes, key=lambda item: item.order or 0):
			mix_path = scenes_dir / f"{scene.id}_mix.wav"
			if not mix_path.exists():
				raise FileNotFoundError(f"Missing scene mix: {mix_path}")
			mix_paths.append(mix_path)

		final_audio_path = final_dir / f"{state.meta.project_id}_final.wav"
		self._merger.concatenate(mix_paths, final_audio_path, gap_ms=0)
		state.audio.final_audio_file = str(final_audio_path)

		now = datetime.now(timezone.utc)
		state.meta.updated_at = now
		if state.phases is None:
			state.phases = PhaseStatusMap()
		state.phases.audio = PhaseStatus(
			status="complete", updated_at=now, message="Audio updated"
		)

		return final_audio_path
