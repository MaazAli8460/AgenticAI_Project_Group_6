import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool


def update_all_states(project_id: str, updater_fn) -> bool:
    """Applies the updater_fn to all existing state files for a project."""
    base_dir = Path("data/outputs")
    paths = [
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
            # Delete cached scene assets to force regeneration
            base_dir = Path(f"data/outputs/phase3/{project_id}")
            for p in (base_dir / "images").glob(f"{scene_id}_*.png"):
                p.unlink(missing_ok=True)
            for p in (base_dir / "clips").glob(f"{scene_id}*.*"):
                p.unlink(missing_ok=True)
            for p in (base_dir / "lip_sync").glob(f"{scene_id}*.*"):
                p.unlink(missing_ok=True)
            return f"Successfully updated scene {scene_id} background to '{new_setting}' across all phases."

        return f"Error: Scene {scene_id} not found."
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
            # Delete cached character assets to force regeneration
            base_dir = Path(f"data/outputs/phase3/{project_id}")
            for p in (base_dir / "images").glob(f"*_{character_id}.png"):
                p.unlink(missing_ok=True)
            for p in (base_dir / "clips").glob("*.*"):
                p.unlink(missing_ok=True)
            for p in (base_dir / "lip_sync").glob("*.*"):
                p.unlink(missing_ok=True)
            return f"Successfully updated character {character_id} appearance across all phases."

        return f"Error: Character {character_id} not found."
    except Exception as e:
        return f"Error updating character: {str(e)}"


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
            # Delete cached BGM to ensure it gets regenerated
            base_dir = Path(f"data/outputs/phase2/{project_id}")
            (base_dir / "bgm" / f"{scene_id}_bgm.wav").unlink(missing_ok=True)
            (base_dir / "scenes" / f"{scene_id}_mix.wav").unlink(missing_ok=True)
            
            # Since the audio changes, we should also delete the final video clip so Phase 3 remuxes it
            (Path(f"data/outputs/phase3/{project_id}/final/{project_id}_final.mp4")).unlink(missing_ok=True)
            return f"Successfully updated scene {scene_id} BGM to '{new_mood}' across all phases."

        return f"Error: Scene {scene_id} not found."
    except Exception as e:
        return f"Error updating BGM: {str(e)}"


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
    update_character_visual,
    update_voice_tone,
    update_scene_bgm,
    run_pipeline_phase
]
