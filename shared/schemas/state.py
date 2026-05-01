from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ProjectMeta(BaseModel):
    project_id: str
    prompt: str
    created_at: datetime
    version: int = Field(ge=1)
    updated_at: Optional[datetime] = None
    language: Optional[str] = None
    seed: Optional[int] = None
    total_duration_s: Optional[float] = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)


class Story(BaseModel):
    logline: str
    synopsis: str
    genre: Optional[str] = None
    theme: Optional[str] = None
    tone: Optional[str] = None
    style: Optional[str] = None
    target_duration_s: Optional[float] = Field(default=None, ge=0)


class VoiceProfile(BaseModel):
    style: str
    gender: Optional[str] = None
    age: Optional[str] = None
    accent: Optional[str] = None
    model: Optional[str] = None
    params: dict[str, object] = Field(default_factory=dict)


class VisualProfile(BaseModel):
    description: str
    style: Optional[str] = None
    palette: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class Character(BaseModel):
    id: str
    name: str
    role: str
    description: str
    voice: VoiceProfile
    visual: VisualProfile


class DialogueLine(BaseModel):
    id: str
    character_id: str
    text: str
    emotion: Optional[str] = None
    direction: Optional[str] = None
    start_ms: Optional[int] = Field(default=None, ge=0)
    end_ms: Optional[int] = Field(default=None, ge=0)


class Scene(BaseModel):
    id: str
    title: str
    description: str
    setting: str
    mood: str
    duration_s: float = Field(ge=0)
    visual_prompt: str
    dialogue: list[DialogueLine] = Field(default_factory=list)
    style: Optional[str] = None
    camera: Optional[str] = None
    motion: Optional[str] = None
    bgm_mood: Optional[str] = None
    bgm_style: Optional[str] = None


class TimingEntry(BaseModel):
    scene_id: str
    line_id: str
    audio_file: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    character_id: Optional[str] = None


class AudioLine(BaseModel):
    line_id: str
    scene_id: str
    character_id: str
    audio_file: str
    start_ms: Optional[int] = Field(default=None, ge=0)
    end_ms: Optional[int] = Field(default=None, ge=0)
    voice_params: dict[str, object] = Field(default_factory=dict)


class BgmTrack(BaseModel):
    scene_id: str
    audio_file: str
    start_ms: Optional[int] = Field(default=None, ge=0)
    end_ms: Optional[int] = Field(default=None, ge=0)
    mood: Optional[str] = None
    style: Optional[str] = None


class AudioState(BaseModel):
    timing_manifest: list[TimingEntry] = Field(default_factory=list)
    lines: list[AudioLine] = Field(default_factory=list)
    bgm_tracks: list[BgmTrack] = Field(default_factory=list)


class SceneAsset(BaseModel):
    scene_id: str
    image_files: list[str] = Field(default_factory=list)
    clip_file: Optional[str] = None
    subtitle_file: Optional[str] = None
    transition: Optional[str] = None


class VideoState(BaseModel):
    scene_assets: list[SceneAsset] = Field(default_factory=list)
    final_video_file: Optional[str] = None
    subtitle_file: Optional[str] = None


class VersionRecord(BaseModel):
    version: int = Field(ge=1)
    timestamp: datetime
    summary: str
    state_path: str
    asset_paths: list[str] = Field(default_factory=list)
    changed_fields: list[str] = Field(default_factory=list)


class EditHistory(BaseModel):
    versions: list[VersionRecord] = Field(default_factory=list)


class PipelineState(BaseModel):
    meta: ProjectMeta
    story: Story
    scenes: list[Scene] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    audio: Optional[AudioState] = None
    video: Optional[VideoState] = None
    edits: Optional[EditHistory] = None
