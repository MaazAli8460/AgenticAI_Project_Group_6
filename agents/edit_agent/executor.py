import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from agents.audio_agent.agent import AudioAgent
from agents.video_agent.agent import VideoAgent
from mcp.tools.video_tools.compositor_tool import CompositorTool
from mcp.tools.video_tools.ffmpeg_tool import FfmpegTool
from shared.schemas.state import PipelineState


def update_all_states(project_id: str, updater_fn) -> bool:
    """Applies the updater_fn to all existing state files for a project."""
    base_dir = Path("data/outputs")
    paths = [
        base_dir / "phase3_lip_sync" / project_id / "state.json",
        base_dir / "phase3" / project_id / "state.json",
        base_dir / "phase2" / project_id / "state.json",
        base_dir / "phase1" / f"{project_id}.json"
    ]
    success = False
    for p in paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if updater_fn(state):
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)
                    success = True
            except Exception as e:
                print(f"Error updating state at {p}: {e}")
    return success


def _phase3_output_dir(project_id: Optional[str] = None) -> Path:
    base_dir = Path("data/outputs")
    if project_id:
        try:
            state_path = get_latest_state_path(project_id)
            if "phase3_lip_sync" in state_path.parts:
                return base_dir / "phase3_lip_sync"
            if "phase3" in state_path.parts:
                return base_dir / "phase3"
        except Exception:
            pass
    enabled = os.getenv("LIP_SYNC_ENABLED", "0").lower() not in {"0", "false", "no"}
    return base_dir / ("phase3_lip_sync" if enabled else "phase3")


def _load_pipeline_state(project_id: str) -> PipelineState:
    state_path = get_latest_state_path(project_id)
    return PipelineState.model_validate_json(state_path.read_text(encoding="utf-8"))


def _save_state_all_phases(project_id: str, state: PipelineState) -> None:
    base_dir = Path("data/outputs")
    targets = [
        base_dir / "phase3_lip_sync" / project_id / "state.json",
        base_dir / "phase3" / project_id / "state.json",
        base_dir / "phase2" / project_id / "state.json",
        base_dir / "phase1" / f"{project_id}.json",
    ]
    payload = json.dumps(state.model_dump(mode="json"), indent=2)
    wrote = False
    for target in targets:
        if target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8")
            wrote = True
    if not wrote:
        target = base_dir / "phase1" / f"{project_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")


def _remux_video_with_audio(project_id: str, state: PipelineState) -> str:
    if not state.video or not state.audio or not state.audio.final_audio_file:
        return "No video or audio to remux."
    output_dir = _phase3_output_dir(project_id)
    final_dir = output_dir / project_id / "final"
    visual_path = final_dir / f"{project_id}_visual.mp4"
    if not visual_path.exists():
        return "Visual video not found; re-run Phase 3 to apply audio."

    audio_path = Path(state.audio.final_audio_file)
    if not audio_path.exists():
        return "Audio output missing; re-run Phase 2 to apply audio."

    try:
        ffmpeg = FfmpegTool()
        compositor = CompositorTool(ffmpeg)
        final_path = final_dir / f"{project_id}_final.mp4"
        compositor.mux_audio(visual_path, audio_path, final_path)

        if state.video.subtitle_file:
            subtitle_path = Path(state.video.subtitle_file)
            if subtitle_path.exists():
                burned_path = final_dir / f"{project_id}_subtitled.mp4"
                try:
                    compositor.burn_subtitles(final_path, subtitle_path, burned_path)
                    state.video.final_video_file = str(burned_path)
                except Exception:
                    state.video.final_video_file = str(final_path)
            else:
                state.video.final_video_file = str(final_path)
        else:
            state.video.final_video_file = str(final_path)

        _save_state_all_phases(project_id, state)
        return "Video remuxed with updated audio."
    except Exception as exc:
        return f"Audio updated, but remux failed: {exc}"


def _scene_ids_for_character(state: PipelineState, character_id: str) -> list[str]:
    return [scene.id for scene in state.scenes if character_id in scene.character_ids]


def _regenerate_scenes(
    project_id: str,
    scene_ids: list[str],
    lip_sync: bool,
) -> str:
    if not scene_ids:
        return json.dumps({"error": "scene_not_found", "scene_id": None})

    try:
        state = _load_pipeline_state(project_id)
        output_dir = _phase3_output_dir(project_id)
        project_root = output_dir / project_id
        if not project_root.exists():
            return "Phase 3 outputs not found. Run Phase 3 once before editing visuals."

        project_root = output_dir / project_id
        images_dir = project_root / "images"
        clips_dir = project_root / "clips"
        lip_sync_dir = project_root / "lip_sync"

        agent = VideoAgent()
        for scene_id in scene_ids:
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for path in images_dir.glob(f"{scene_id}_*{pattern.lstrip('*')}"):
                    path.unlink(missing_ok=True)
            for path in clips_dir.glob(f"{scene_id}_seg_*.*"):
                path.unlink(missing_ok=True)
            (clips_dir / f"{scene_id}.mp4").unlink(missing_ok=True)
            for path in lip_sync_dir.glob(f"{scene_id}_*.*"):
                path.unlink(missing_ok=True)

            agent.generate_scene(state, output_dir, scene_id, lip_sync_override=lip_sync)
        agent.rebuild_final_video(state, output_dir, burn_subtitles=None)

        output_path = output_dir / project_id / "state.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8")
        _save_state_all_phases(project_id, state)
        return f"Regenerated scenes: {', '.join(scene_ids)}"
    except Exception as exc:
        return f"Error regenerating scenes: {exc}"


def get_latest_state_path(project_id: str) -> Path:
    """Return the most recent state.json for a project across phases."""
    base_dir = Path("data/outputs")
    candidates = [
        base_dir / "phase3_lip_sync" / project_id / "state.json",
        base_dir / "phase3" / project_id / "state.json",
        base_dir / "phase2" / project_id / "state.json",
        base_dir / "phase1" / f"{project_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No state.json found for project {project_id}")


@tool
def get_project_context(project_id: str) -> str:
    """
    Returns the current project state (characters, scenes, etc.) so you can understand what currently exists.
    Use this if the user asks a generic question or if you need to know character/scene IDs before making an edit.
    """
    try:
        state_path = get_latest_state_path(project_id)
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # We only return the essential metadata to avoid blowing up the context window
        context = {
            "characters": [{"id": c.get("id"), "name": c.get("name"), "visual": c.get("visual")} for c in state.get("characters", [])],
            "scenes": [{"id": s.get("id"), "setting": s.get("setting"), "characters": s.get("character_ids")} for s in state.get("scenes", [])]
        }
        return json.dumps(context, indent=2)
    except Exception as e:
        return f"Error getting context: {e}"

@tool
def get_asset_path(project_id: str, asset_type: str, target_id: str) -> str:
    """
    Returns the local path to a generated asset (like a character image or scene clip) so you can show it to the user.
    asset_type should be 'character_image' or 'scene_clip'.
    target_id should be the character_id or scene_id.
    """
    base_dir = Path("data/outputs/phase3") / project_id
    if asset_type == "character_image":
        # Look for images in the images directory
        images_dir = base_dir / "images"
        if not images_dir.exists(): return "Images not generated yet."
        for img in images_dir.glob(f"*_{target_id}.png"):
            return f"Found image: {img.absolute()}"
        return f"Image for character {target_id} not found."
    return "Asset type not supported."

@tool
def update_scene_background(project_id: str, scene_id: str, new_setting: str) -> str:
    """
    Updates the setting/background of a specific scene in the project state.
    Call this when the user wants to change a scene's visual background.
    """
    try:
        def updater(state):
            updated = False
            for scene in state.get("scenes", []):
                if scene.get("id") == scene_id:
                    scene["setting"] = new_setting
                    scene["visual_prompt"] = f"{new_setting}. {scene.get('visual_prompt', '')}"
                    updated = True
            return updated

        if update_all_states(project_id, updater):
            # Regenerate only the targeted scene; skip lip-sync for background edits.
            result = _regenerate_scenes(project_id, [scene_id], lip_sync=False)
            return f"Updated scene {scene_id} background to '{new_setting}'. {result}"

        return json.dumps({"error": "scene_not_found", "scene_id": scene_id})
    except Exception as e:
        return f"Error updating scene: {str(e)}"


@tool
def update_character_visual(project_id: str, character_id: str, new_visual: str) -> str:
    """
    Updates a character's visual description. 
    Call this when the user wants to change how a character looks.
    """
    try:
        def updater(state):
            updated = False
            for char in state.get("characters", []):
                if char.get("id") == character_id:
                    if "visual" not in char: char["visual"] = {}
                    char["visual"]["description"] = new_visual
                    updated = True
            return updated

        if update_all_states(project_id, updater):
            state = _load_pipeline_state(project_id)
            scene_ids = _scene_ids_for_character(state, character_id)
            if not scene_ids:
                return json.dumps({"error": "character_not_found", "character_id": character_id})
            lip_sync = os.getenv("LIP_SYNC_ENABLED", "0").lower() not in {"0", "false", "no"}
            result = _regenerate_scenes(project_id, scene_ids, lip_sync=lip_sync)
            return f"Updated character {character_id} appearance. {result}"

        return json.dumps({"error": "character_not_found", "character_id": character_id})
    except Exception as e:
        return f"Error updating character: {str(e)}"


@tool
def update_scene_character_visual(
    project_id: str,
    scene_id: str,
    character_id: str,
    new_visual: str,
) -> str:
    """
    Updates a character's visual description for a single scene only.
    Uses scene-level overrides so other scenes are unchanged.
    """
    try:
        def updater(state):
            updated = False
            for scene in state.get("scenes", []):
                if scene.get("id") == scene_id:
                    overrides = scene.get("character_overrides") or {}
                    overrides[character_id] = new_visual
                    scene["character_overrides"] = overrides
                    updated = True
            return updated

        if update_all_states(project_id, updater):
            lip_sync = os.getenv("LIP_SYNC_ENABLED", "0").lower() not in {"0", "false", "no"}
            result = _regenerate_scenes(project_id, [scene_id], lip_sync=lip_sync)
            return f"Updated {character_id} appearance in {scene_id}. {result}"

        return json.dumps({"error": "scene_not_found", "scene_id": scene_id})
    except Exception as e:
        return f"Error updating scene character: {str(e)}"


@tool
def update_voice_tone(project_id: str, character_id: str, new_tone: str) -> str:
    """
    Updates a character's voice tone or style.
    Call this when the user wants to change how a character sounds.
    """
    try:
        def updater(state):
            updated = False
            for char in state.get("characters", []):
                if char.get("id") == character_id:
                    if "voice" not in char: char["voice"] = {}
                    char["voice"]["style"] = new_tone
                    updated = True
            return updated

        if update_all_states(project_id, updater):
            # For voice updates, we'd need to delete audio caches from phase2
            # and phase3_lip_sync, but for now just updating the state is good.
            return f"Successfully updated character {character_id} voice tone across all phases."

        return f"Error: Character {character_id} not found."
    except Exception as e:
        return f"Error updating voice: {str(e)}"

@tool
def update_scene_bgm(project_id: str, scene_id: str, new_mood: str) -> str:
    """
    Updates the background music (BGM) mood for a specific scene.
    Call this when the user wants to change the music or audio vibe of a scene.
    """
    try:
        def updater(state):
            updated = False
            for scene in state.get("scenes", []):
                if scene.get("id") == scene_id:
                    scene["bgm_mood"] = new_mood
                    updated = True
            return updated

        if update_all_states(project_id, updater):
            output_dir = Path("data/outputs/phase2")
            audio_agent = AudioAgent()
            state = _load_pipeline_state(project_id)
            audio_agent.regenerate_scene_bgm(state, output_dir, scene_id)
            audio_agent.rebuild_final_audio(state, output_dir)
            _save_state_all_phases(project_id, state)
            remux_msg = _remux_video_with_audio(project_id, state)
            return f"Updated scene {scene_id} BGM to '{new_mood}'. {remux_msg}"

        return json.dumps({"error": "scene_not_found", "scene_id": scene_id})
    except Exception as e:
        return f"Error updating BGM: {str(e)}"


@tool
def update_scene_dialogue(project_id: str, scene_id: str, preserve_timing: bool = True) -> str:
    """
    Regenerates TTS dialogue for a single scene and remixes audio.
    Keeps existing timing by default so video stays in sync.
    """
    try:
        output_dir = Path("data/outputs/phase2")
        audio_agent = AudioAgent()
        state = _load_pipeline_state(project_id)
        audio_agent.regenerate_scene_dialogue(
            state,
            output_dir,
            scene_id,
            preserve_timing=preserve_timing,
        )
        audio_agent.rebuild_final_audio(state, output_dir)
        _save_state_all_phases(project_id, state)
        remux_msg = _remux_video_with_audio(project_id, state)
        return f"Regenerated dialogue for scene {scene_id}. {remux_msg}"
    except Exception as e:
        return f"Error updating scene dialogue: {str(e)}"


@tool
def toggle_subtitles(project_id: str, enabled: bool) -> str:
    """
    Toggle subtitles by selecting the pre-rendered subtitled or non-subtitled MP4.
    Does not re-run Phase 3.
    """
    try:
        state = _load_pipeline_state(project_id)
        output_dir = _phase3_output_dir(project_id)
        final_dir = output_dir / project_id / "final"
        base_video = final_dir / f"{project_id}_final.mp4"
        subtitled_video = final_dir / f"{project_id}_subtitled.mp4"
        target = subtitled_video if enabled else base_video
        if not target.exists():
            return json.dumps({"error": "video_not_found", "path": str(target)})
        if state.video is None:
            from shared.schemas.state import VideoState
            state.video = VideoState()
        state.video.final_video_file = str(target)
        if enabled:
            subtitle_path = output_dir / project_id / "subtitles" / f"{project_id}.srt"
            state.video.subtitle_file = str(subtitle_path) if subtitle_path.exists() else None
        else:
            state.video.subtitle_file = None
        payload = json.dumps(state.model_dump(mode="json"), indent=2)
        try:
            state_path = get_latest_state_path(project_id)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(payload, encoding="utf-8")
        except Exception:
            pass
        _save_state_all_phases(project_id, state)
        return f"Subtitles {'enabled' if enabled else 'disabled'}."
    except Exception as e:
        return f"Error toggling subtitles: {str(e)}"


@tool
def run_pipeline_phase(project_id: str, phase: int) -> str:
    """
    Executes a specific pipeline phase to apply changes.
    - Phase 2: Generates Audio (TTS & BGM). Run this if audio/voice/script changed.
    - Phase 3: Generates Video (Images & Compositing). Run this if visual/scene/character looks changed.
    """
    try:
        script = f"run_phase{phase}.py"
        
        if phase == 2:
            input_file = Path("data/outputs/phase1") / f"{project_id}.json"
        else:
            input_file = Path("data/outputs/phase2") / project_id / "state.json"

        if not input_file.exists():
            return f"Error: Input file {input_file} not found for Phase {phase}."

        cmd = ["python", script, "--input", str(input_file)]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"Successfully executed Phase {phase}.\nOutput: {result.stdout[-200:]}"
    except subprocess.CalledProcessError as e:
        return f"Error running Phase {phase}: {e.stderr}"
    except Exception as e:
        return f"System error running phase: {str(e)}"

# List of all tools to bind to the LLM
EDIT_TOOLS = [
    get_project_context,
    get_asset_path,
    update_scene_background,
    update_scene_character_visual,
    update_character_visual,
    update_voice_tone,
    update_scene_bgm,
    update_scene_dialogue,
    toggle_subtitles,
    run_pipeline_phase
]
