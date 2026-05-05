from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
from uuid import uuid4

from agents.edit_agent.executor import toggle_subtitles
from shared.schemas.state import (
    Character,
    PipelineState,
    ProjectMeta,
    Scene,
    Story,
    VideoState,
    VisualProfile,
    VoiceProfile,
)


def _build_state(project_id: str, base_dir: Path) -> Path:
    now = datetime.now(timezone.utc)
    state = PipelineState(
        meta=ProjectMeta(
            project_id=project_id,
            prompt="test",
            created_at=now,
            updated_at=now,
            version=1,
            schema_version="1.0",
        ),
        story=Story(logline="Test", synopsis="Test"),
        scenes=[
            Scene(
                id="scene_1",
                title="Scene",
                description="Test",
                setting="Room",
                mood="calm",
                duration_s=2.0,
                visual_prompt="A calm room",
                dialogue=[],
                character_ids=["char_1"],
            )
        ],
        characters=[
            Character(
                id="char_1",
                name="Ava",
                role="Protagonist",
                description="Curious astronaut",
                voice=VoiceProfile(style="soft", gender="female"),
                visual=VisualProfile(description="Explorer suit"),
            )
        ],
        video=VideoState(
            final_video_file=str(base_dir / "final" / f"{project_id}_final.mp4"),
            subtitle_file=None,
        ),
    )
    state_path = base_dir / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return state_path


def test_toggle_subtitles_selects_existing_files(tmp_path: Path) -> None:
    project_id = f"test_subtitles_{uuid4().hex[:8]}"
    phase3_dir = Path("data/outputs/phase3") / project_id
    final_dir = phase3_dir / "final"
    subtitles_dir = phase3_dir / "subtitles"
    final_dir.mkdir(parents=True, exist_ok=True)
    subtitles_dir.mkdir(parents=True, exist_ok=True)

    base_video = final_dir / f"{project_id}_final.mp4"
    subtitled_video = final_dir / f"{project_id}_subtitled.mp4"
    base_video.write_text("base")
    subtitled_video.write_text("subtitled")
    (subtitles_dir / f"{project_id}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")

    _build_state(project_id, phase3_dir)

    enabled_msg = toggle_subtitles.invoke({"project_id": project_id, "enabled": True})
    assert "enabled" in enabled_msg.lower()

    state_path = phase3_dir / "state.json"
    content = state_path.read_text(encoding="utf-8")
    data = json.loads(content)
    final_path = Path(data["video"]["final_video_file"]).resolve()
    assert final_path == subtitled_video.resolve()

    disabled_msg = toggle_subtitles.invoke({"project_id": project_id, "enabled": False})
    assert "disabled" in disabled_msg.lower()
    content = state_path.read_text(encoding="utf-8")
    data = json.loads(content)
    final_path = Path(data["video"]["final_video_file"]).resolve()
    assert final_path == base_video.resolve()

    shutil.rmtree(phase3_dir, ignore_errors=True)
    shutil.rmtree(Path("data/outputs/phase3_lip_sync") / project_id, ignore_errors=True)
