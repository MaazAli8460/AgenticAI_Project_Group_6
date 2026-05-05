import sys
from pathlib import Path
from agents.edit_agent.executor import (
    update_scene_bgm,
    update_voice_tone,
    run_pipeline_phase
)

project_id = "proj_601cadb0"

print("--- Testing BGM Change ---")
print("Setting scene_1 BGM mood to 'scary'...")
print(update_scene_bgm.invoke({
    "project_id": project_id, 
    "scene_id": "scene_1", 
    "new_mood": "scary"
}))

# Mission Control voice tone was already set to 'Panicked and urgent' in the last test
# but we will set Maya's voice to 'Confident and booming' to test ElevenLabs
print("\n--- Testing Audio Voice Tone Change ---")
print("Setting Maya (char_1) voice to 'Confident and booming'...")
print(update_voice_tone.invoke({
    "project_id": project_id, 
    "character_id": "char_1", 
    "new_tone": "Confident and booming"
}))

print("\n--- Running Phase 2 (Audio & BGM Regeneration) ---")
print(run_pipeline_phase.invoke({"project_id": project_id, "phase": 2}))

print("\n--- Running Phase 3 (Video & Lip Sync Regeneration) ---")
print(run_pipeline_phase.invoke({"project_id": project_id, "phase": 3}))

print("\nDone! Check data/outputs/phase3/proj_601cadb0/final/proj_601cadb0_final.mp4")
