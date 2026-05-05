import json
from pathlib import Path

from state_manager.state_manager import StateManager


def test_state_manager(tmp_path: Path):
    base_dir = tmp_path / "outputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup mock project assets
    project_id = "test_proj_123"
    phase1_file = base_dir / "phase1" / f"{project_id}.json"
    phase1_file.parent.mkdir(parents=True, exist_ok=True)
    phase1_file.write_text('{"foo": "bar"}')
    
    phase2_dir = base_dir / "phase2" / project_id
    phase2_dir.mkdir(parents=True, exist_ok=True)
    (phase2_dir / "audio.wav").write_text("mock audio")
    
    state_json = {"meta": {"version": 1}, "story": "Initial Story"}
    asset_paths = [phase1_file, phase2_dir]
    
    manager = StateManager(project_id, base_dir)
    
    # 2. Snapshot v1
    print("Taking snapshot v1...")
    manager.snapshot("v1", state_json, asset_paths, "Initial generation")
    assert manager.previous_version() is None
    
    # 3. Modify state and assets
    state_json["story"] = "Edited Story"
    (phase2_dir / "audio.wav").write_text("edited audio")
    
    # 4. Snapshot v2
    print("Taking snapshot v2...")
    manager.snapshot("v2", state_json, asset_paths, "Edited audio and story")
    assert manager.previous_version() == "v1"
    
    # 5. Check history
    print("\nHistory:")
    print(json.dumps(manager.history(), indent=2))
    
    # 6. Revert to v1
    print("\nReverting to v1...")
    restored_state = manager.revert("v1")
    print(f"Restored state story: {restored_state['story']}")
    
    # Check if file was restored
    restored_audio = (phase2_dir / "audio.wav").read_text()
    print(f"Restored audio content: {restored_audio}")
    
    assert restored_state["story"] == "Initial Story"
    assert restored_audio == "mock audio"
    print("\n✅ StateManager works perfectly!")

if __name__ == "__main__":
    test_state_manager(Path("data/outputs"))
