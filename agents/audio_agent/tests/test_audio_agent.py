from datetime import datetime, timezone
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.audio_agent.agent import AudioAgent
from shared.schemas.state import (
    Character,
    DialogueLine,
    PipelineState,
    ProjectMeta,
    Scene,
    Story,
    VisualProfile,
    VoiceProfile,
)


def build_state() -> PipelineState:
    now = datetime.now(timezone.utc)
    characters = [
        Character(
            id="char_1",
            name="Ava",
            role="Protagonist",
            description="Curious astronaut",
            voice=VoiceProfile(style="soft", gender="female"),
            visual=VisualProfile(description="Explorer suit"),
        ),
        Character(
            id="char_2",
            name="Mission Control",
            role="Supporting",
            description="Guiding voice",
            voice=VoiceProfile(style="calm", gender="male"),
            visual=VisualProfile(description="Face on a screen"),
        ),
    ]
    scenes = [
        Scene(
            id="scene_1",
            title="Discovery",
            description="Ava finds the hidden ocean",
            setting="Mars surface",
            mood="serene",
            duration_s=5.0,
            visual_prompt="Ava at the edge of a glowing sea",
            dialogue=[
                DialogueLine(
                    id="line_1",
                    character_id="char_1",
                    text="It cannot be real.",
                )
            ],
            character_ids=["char_1"],
        ),
        Scene(
            id="scene_2",
            title="Dilemma",
            description="Ava debates the choice",
            setting="Spacecraft cabin",
            mood="contemplative",
            duration_s=7.0,
            visual_prompt="Ava in a dim cabin, screens glowing",
            dialogue=[
                DialogueLine(
                    id="line_2",
                    character_id="char_1",
                    text="What do I do now?",
                ),
                DialogueLine(
                    id="line_3",
                    character_id="char_2",
                    text="Report your findings.",
                ),
            ],
            character_ids=["char_1", "char_2"],
        ),
    ]

    return PipelineState(
        meta=ProjectMeta(
            project_id="proj_test",
            prompt="Test prompt",
            created_at=now,
            updated_at=now,
            version=1,
            schema_version="1.0",
        ),
        story=Story(logline="Test", synopsis="Test"),
        scenes=scenes,
        characters=characters,
    )


def test_audio_agent_generates_outputs(tmp_path: Path) -> None:
    os.environ["TTS_PROVIDER"] = "none"
    state = build_state()
    agent = AudioAgent()
    updated = agent.generate(state, output_dir=tmp_path)

    assert updated.audio is not None
    assert updated.audio.timing_manifest
    assert len(updated.audio.lines) == 3
    assert updated.audio.final_audio_file is not None

    for line in updated.audio.lines:
        assert Path(line.audio_file).exists()

    assert Path(updated.audio.final_audio_file).exists()
    assert updated.phases is not None
    assert updated.phases.audio is not None
    assert updated.phases.audio.status == "complete"


def test_audio_agent_regenerates_scene_dialogue(tmp_path: Path) -> None:
    os.environ["TTS_PROVIDER"] = "none"
    state = build_state()
    agent = AudioAgent()
    updated = agent.generate(state, output_dir=tmp_path)

    assert updated.audio is not None
    original_entries = {
        entry.line_id: (entry.start_ms, entry.end_ms)
        for entry in updated.audio.timing_manifest
        if entry.scene_id == "scene_1"
    }

    updated = agent.regenerate_scene_dialogue(updated, output_dir=tmp_path, scene_id="scene_1")
    assert updated.audio is not None
    new_entries = {
        entry.line_id: (entry.start_ms, entry.end_ms)
        for entry in updated.audio.timing_manifest
        if entry.scene_id == "scene_1"
    }

    assert new_entries == original_entries

    project_root = tmp_path / state.meta.project_id
    assert (project_root / "scenes" / "scene_1_dialogue.wav").exists()
    assert (project_root / "scenes" / "scene_1_mix.wav").exists()
