import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class StateManager:
    """
    Manages snapshots, history, and undo functionality for a project.
    Snapshots are saved as an append-only log with copied assets, ensuring
    full revertibility.
    """

    def __init__(self, project_id: str, base_dir: Union[Path, str]):
        self.project_id = project_id
        self.base_dir = Path(base_dir).resolve()
        self.snapshots_dir = self.base_dir / "snapshots" / self.project_id
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.snapshots_dir / "history.json"

        if not self.history_file.exists():
            self._save_history([])

    def _load_history(self) -> List[Dict[str, Any]]:
        with open(self.history_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def snapshot(
        self,
        version: str,
        state_json: dict,
        asset_paths: List[Union[Path, str]],
        diff_summary: str = "",
    ) -> None:
        """
        Takes a snapshot of the current state and specified assets.

        Args:
            version: Unique version identifier (e.g., "v1", "v2").
            state_json: The full PipelineState as a dictionary.
            asset_paths: List of paths (files or directories) to backup.
            diff_summary: Natural language summary of what changed.
        """
        version_dir = self.snapshots_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        version_dir.mkdir(parents=True)

        # 1. Save state.json
        state_path = version_dir / "state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_json, f, indent=2)

        # 2. Backup assets
        # We store them in version_dir/assets/... maintaining their relative paths
        # so they can be accurately restored.
        snap_assets_dir = version_dir / "assets"
        
        for path in asset_paths:
            p = Path(path).resolve()
            if not p.exists():
                continue

            try:
                # Try to get path relative to base_dir (e.g. data/outputs)
                rel_path = p.relative_to(self.base_dir)
            except ValueError:
                # If path is outside base_dir, just use its name
                rel_path = Path(p.name)

            dest = snap_assets_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            if p.is_dir():
                shutil.copytree(p, dest)
            else:
                shutil.copy2(p, dest)

        # 3. Append to history
        history = self._load_history()
        # Remove existing if overwriting version
        history = [h for h in history if h["version"] != version]
        
        history.append({
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "diff_summary": diff_summary,
        })
        self._save_history(history)

    def revert(self, version: str) -> dict:
        """
        Restores the assets and returns the state_json for the requested version.
        
        Args:
            version: The version to revert to.
            
        Returns:
            The parsed state.json dictionary of that version.
        """
        version_dir = self.snapshots_dir / version
        if not version_dir.exists():
            raise ValueError(f"Snapshot version '{version}' not found.")

        # 1. Load the backed-up state
        state_path = version_dir / "state.json"
        with open(state_path, "r", encoding="utf-8") as f:
            state_json = json.load(f)

        # 2. Restore assets
        snap_assets_dir = version_dir / "assets"
        if snap_assets_dir.exists():
            # Iterate through everything in the snapshot assets dir
            for snap_path in snap_assets_dir.rglob("*"):
                if snap_path.is_file():
                    rel_path = snap_path.relative_to(snap_assets_dir)
                    dest_path = self.base_dir / rel_path
                    
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snap_path, dest_path)

        return state_json

    def history(self) -> List[Dict[str, Any]]:
        """Returns the chronological list of all saved snapshots."""
        return self._load_history()

    def previous_version(self) -> Optional[str]:
        """Return the previous snapshot version (vN-1), or None if unavailable."""
        history = self._load_history()
        if len(history) < 2:
            return None
        return history[-2].get("version")
