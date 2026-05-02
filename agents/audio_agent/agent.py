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

		current_ms = 0
		scenes = sorted(state.scenes, key=lambda item: item.order or 0)
		for scene in scenes:
			scene_line_paths: list[Path] = []
			line_items: list[tuple[Path, int]] = []
			scene_start_ms = current_ms
			last_line_end = scene_start_ms

			for line_index, line in enumerate(scene.dialogue, start=1):
				character = character_map.get(line.character_id, state.characters[0])
				file_name = line.id or f"line_{scene.id}_{line_index}"
				line_path = lines_dir / f"{file_name}.wav"
				duration_ms = self._tts.synthesize(line.text, character.voice, line_path)
				start_ms = current_ms
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

				last_line_end = end_ms
				current_ms = end_ms + self._gap_ms
				line_items.append((line_path, start_ms - scene_start_ms))

			scene_duration_ms = max(
				int(scene.duration_s * 1000),
				int(last_line_end - scene_start_ms),
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
			current_ms = max(current_ms, scene_end_ms + self._gap_ms)

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
