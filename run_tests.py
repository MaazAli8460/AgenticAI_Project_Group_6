import sys
from pathlib import Path
from agents.edit_agent.executor import (
    update_scene_background,
    update_voice_tone,
    run_pipeline_phase
)

project_id = "proj_601cadb0"

print("--- Testing Background Change ---")
print("Setting scene_1 background to 'Mars surface with red dust and rovers'...")
print(update_scene_background.invoke({
    "project_id": project_id, 
    "scene_id": "scene_1", 
    "new_setting": "Mars surface with red dust and rovers"
}))

print("\n--- Testing Audio Voice Tone Change ---")
print("Setting Mission Control (char_2) voice to 'Panicked and urgent'...")
print(update_voice_tone.invoke({
    "project_id": project_id, 
    "character_id": "char_2", 
    "new_tone": "Panicked and urgent"
}))

print("\n--- Running Phase 2 (Audio Regeneration) ---")
print(run_pipeline_phase.invoke({"project_id": project_id, "phase": 2}))

print("\n--- Running Phase 3 (Video & Lip Sync Regeneration) ---")
print(run_pipeline_phase.invoke({"project_id": project_id, "phase": 3}))

print("\nDone! Check data/outputs/phase3/proj_601cadb0/final/proj_601cadb0_final.mp4")
