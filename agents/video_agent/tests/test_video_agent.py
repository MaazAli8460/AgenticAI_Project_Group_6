from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.video_agent.agent import VideoAgent
from mcp.tools.video_tools.compositor_tool import CompositorTool
from mcp.tools.video_tools.wav2lip_tool import Wav2LipError
from shared.schemas.state import (
    AudioLine,
    AudioState,
    BgmTrack,
    Character,
    DialogueLine,
    PipelineState,
    ProjectMeta,
    Scene,
    Story,
    TimingEntry,
    VisualProfile,
    VoiceProfile,
)
from mcp.tools.audio_tools.audio_utils import generate_silence_samples, write_wav


class FakeImageGenTool:
    def generate(self, prompt, output_path, width, height, seed=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
                "530000000a49444154789c6360000002000155c2210b0000000049454e44ae"
                "426082"
            )
        )
        return output_path


class FakeFfmpegTool:
    def __init__(self) -> None:
        self.image_to_clip_calls: list[tuple[Path, float]] = []
        self.normalize_clip_calls: list[Path] = []

    def image_to_clip(self, image_path, output_path, duration_s, width, height, fps, effect="zoom_in"):
        self.image_to_clip_calls.append((image_path, duration_s))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")
        return output_path

    def concat_clips(self, list_file, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")
        return output_path

    def mux_audio(self, video_path, audio_path, output_path):
        output_path.write_bytes(b"fake")
        return output_path

    def add_subtitles(self, video_path, subtitle_path, output_path):
        output_path.write_bytes(b"fake")
        return output_path

    def normalize_clip(self, input_path, output_path, width, height, fps):
        self.normalize_clip_calls.append(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-normalized")
        return output_path


class FakeWav2LipTool:
    def __init__(self, availability: str | None = None, raise_on_call: bool = False) -> None:
        self._availability = availability
        self._raise = raise_on_call
        self.calls: list[tuple[Path, Path, Path]] = []

    def is_available(self) -> bool:
        return self._availability is None

    def availability_reason(self) -> str | None:
        return self._availability

    def lip_sync(self, face_path: Path, audio_path: Path, output_path: Path) -> Path:
        self.calls.append((face_path, audio_path, output_path))
        if self._raise:
            raise Wav2LipError("synthetic failure")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-synced")
        return output_path


def build_state(tmp_path: Path) -> PipelineState:
    now = datetime.now(timezone.utc)
    audio_path = tmp_path / "audio.wav"
    samples = generate_silence_samples(2000)
    write_wav(audio_path, samples)

    characters = [
        Character(
            id="char_1",
            name="Ava",
            role="Protagonist",
            description="Curious astronaut",
            voice=VoiceProfile(style="soft", gender="female"),
            visual=VisualProfile(description="Explorer suit"),
        )
    ]
    scenes = [
        Scene(
            id="scene_1",
            title="Discovery",
            description="Ava finds the ocean",
            setting="Mars surface",
            mood="serene",
            duration_s=2.0,
            visual_prompt="Ava at the edge of a glowing sea",
            dialogue=[
                DialogueLine(
                    id="line_1",
                    character_id="char_1",
                    text="It cannot be real.",
                )
            ],
            character_ids=["char_1"],
        )
    ]
    audio_state = AudioState(
        timing_manifest=[TimingEntry(scene_id="scene_1", line_id="line_1", audio_file=str(audio_path), start_ms=0, end_ms=1500)],
        lines=[AudioLine(line_id="line_1", scene_id="scene_1", character_id="char_1", audio_file=str(audio_path), text="It cannot be real.")],
        bgm_tracks=[BgmTrack(scene_id="scene_1", audio_file=str(audio_path), start_ms=0, end_ms=2000, mood="atmospheric")],
        final_audio_file=str(audio_path),
    )

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
        audio=audio_state,
    )


def build_state_multi_line(tmp_path: Path) -> PipelineState:
    now = datetime.now(timezone.utc)
    audio_path = tmp_path / "audio.wav"
    samples = generate_silence_samples(3000)
    write_wav(audio_path, samples)

    characters = [
        Character(
            id="char_1",
            name="Ava",
            role="Protagonist",
            description="Curious astronaut",
            voice=VoiceProfile(style="soft", gender="female"),
            visual=VisualProfile(description="Explorer suit"),
        )
    ]
    scenes = [
        Scene(
            id="scene_1",
            title="Discovery",
            description="Ava reflects on the signal",
            setting="Mars surface",
            mood="serene",
            duration_s=3.0,
            visual_prompt="Ava in a quiet crater",
            dialogue=[
                DialogueLine(
                    id="line_1",
                    character_id="char_1",
                    text="We are not alone.",
                ),
                DialogueLine(
                    id="line_2",
                    character_id="char_1",
                    text="I can feel it.",
                ),
            ],
            character_ids=["char_1"],
        )
    ]
    audio_state = AudioState(
        timing_manifest=[
            TimingEntry(
                scene_id="scene_1",
                line_id="line_1",
                audio_file=str(audio_path),
                start_ms=0,
                end_ms=800,
                character_id="char_1",
            ),
            TimingEntry(
                scene_id="scene_1",
                line_id="line_2",
                audio_file=str(audio_path),
                start_ms=1000,
                end_ms=1800,
                character_id="char_1",
            ),
        ],
        lines=[
            AudioLine(
                line_id="line_1",
                scene_id="scene_1",
                character_id="char_1",
                audio_file=str(audio_path),
                text="We are not alone.",
                start_ms=0,
                end_ms=800,
            ),
            AudioLine(
                line_id="line_2",
                scene_id="scene_1",
                character_id="char_1",
                audio_file=str(audio_path),
                text="I can feel it.",
                start_ms=1000,
                end_ms=1800,
            ),
        ],
        bgm_tracks=[
            BgmTrack(
                scene_id="scene_1",
                audio_file=str(audio_path),
                start_ms=0,
                end_ms=3000,
                mood="atmospheric",
            )
        ],
        final_audio_file=str(audio_path),
    )

    return PipelineState(
        meta=ProjectMeta(
            project_id="proj_test_multi",
            prompt="Test prompt",
            created_at=now,
            updated_at=now,
            version=1,
            schema_version="1.0",
        ),
        story=Story(logline="Test", synopsis="Test"),
        scenes=scenes,
        characters=characters,
        audio=audio_state,
    )


def test_video_agent_generates_outputs(tmp_path: Path) -> None:
    state = build_state(tmp_path)
    ffmpeg = FakeFfmpegTool()
    agent = VideoAgent(
        image_tool=FakeImageGenTool(),
        ffmpeg_tool=ffmpeg,
        compositor_tool=CompositorTool(ffmpeg),
        subtitles=False,
        lip_sync_enabled=False,
    )
    updated = agent.generate(state, output_dir=tmp_path)

    assert updated.video is not None
    assert updated.video.final_video_file is not None
    assert Path(updated.video.final_video_file).exists()
    assert updated.video.scene_assets


def test_video_agent_groups_single_speaker_segments(tmp_path: Path) -> None:
    state = build_state_multi_line(tmp_path)
    ffmpeg = FakeFfmpegTool()
    agent = VideoAgent(
        image_tool=FakeImageGenTool(),
        ffmpeg_tool=ffmpeg,
        compositor_tool=CompositorTool(ffmpeg),
        subtitles=False,
        lip_sync_enabled=False,
    )
    agent.generate(state, output_dir=tmp_path)

    assert len(ffmpeg.image_to_clip_calls) == 1
    _, duration_s = ffmpeg.image_to_clip_calls[0]
    assert abs(duration_s - 3.0) < 0.05


def test_video_agent_lip_syncs_dialogue_segment(tmp_path: Path) -> None:
    state = build_state_multi_line(tmp_path)
    ffmpeg = FakeFfmpegTool()
    wav2lip = FakeWav2LipTool()
    agent = VideoAgent(
        image_tool=FakeImageGenTool(),
        ffmpeg_tool=ffmpeg,
        compositor_tool=CompositorTool(ffmpeg),
        lip_sync_tool=wav2lip,
        subtitles=False,
        lip_sync_enabled=True,
    )
    agent.generate(state, output_dir=tmp_path)

    assert len(wav2lip.calls) == 1
    face_path, audio_path, synced_path = wav2lip.calls[0]
    assert face_path.exists() and face_path.read_bytes() == b"fake"
    assert audio_path.exists() and audio_path.suffix == ".wav"
    assert synced_path.exists() and synced_path.read_bytes() == b"fake-synced"
    assert ffmpeg.normalize_clip_calls == [synced_path]

    manifest_path = tmp_path / state.meta.project_id / "prompts.json"
    assert manifest_path.exists()
    import json as _json

    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_meta = manifest["scene_1"]
    assert scene_meta["lip_sync_enabled"] is True
    assert scene_meta["lip_sync_segments"]
    assert scene_meta["lip_sync_segments"][0]["status"] == "lip_synced"


def test_video_agent_falls_back_when_wav2lip_fails(tmp_path: Path) -> None:
    state = build_state_multi_line(tmp_path)
    ffmpeg = FakeFfmpegTool()
    wav2lip = FakeWav2LipTool(raise_on_call=True)
    agent = VideoAgent(
        image_tool=FakeImageGenTool(),
        ffmpeg_tool=ffmpeg,
        compositor_tool=CompositorTool(ffmpeg),
        lip_sync_tool=wav2lip,
        subtitles=False,
        lip_sync_enabled=True,
        lip_sync_strict=False,
    )
    updated = agent.generate(state, output_dir=tmp_path)

    assert len(wav2lip.calls) == 1
    assert ffmpeg.normalize_clip_calls == []
    assert updated.video is not None and updated.video.final_video_file is not None

    import json as _json

    manifest = _json.loads(
        (tmp_path / state.meta.project_id / "prompts.json").read_text(encoding="utf-8")
    )
    statuses = [seg["status"] for seg in manifest["scene_1"]["lip_sync_segments"]]
    assert statuses == ["fallback"]


def test_video_agent_disables_lip_sync_when_wav2lip_unavailable(tmp_path: Path) -> None:
    state = build_state_multi_line(tmp_path)
    ffmpeg = FakeFfmpegTool()
    wav2lip = FakeWav2LipTool(availability="WAV2LIP_DIR is not set")
    agent = VideoAgent(
        image_tool=FakeImageGenTool(),
        ffmpeg_tool=ffmpeg,
        compositor_tool=CompositorTool(ffmpeg),
        lip_sync_tool=wav2lip,
        subtitles=False,
        lip_sync_enabled=True,
        lip_sync_strict=False,
    )
    agent.generate(state, output_dir=tmp_path)

    assert wav2lip.calls == []
    assert ffmpeg.normalize_clip_calls == []


def test_video_agent_strict_mode_raises_when_wav2lip_unavailable(tmp_path: Path) -> None:
    import pytest

    wav2lip = FakeWav2LipTool(availability="WAV2LIP_DIR is not set")
    with pytest.raises(RuntimeError, match="Wav2Lip is unavailable"):
        VideoAgent(
            image_tool=FakeImageGenTool(),
            ffmpeg_tool=FakeFfmpegTool(),
            lip_sync_tool=wav2lip,
            subtitles=False,
            lip_sync_enabled=True,
            lip_sync_strict=True,
        )
