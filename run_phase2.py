from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agents.audio_agent.agent import AudioAgent
from shared.schemas.state import PipelineState


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Phase 2 audio generation")
    parser.add_argument("--input", type=Path, required=True, help="Phase 1 JSON path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the updated PipelineState JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/phase2"),
        help="Folder to store audio outputs and state JSON",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout in addition to saving",
    )
    args = parser.parse_args()

    state = PipelineState.model_validate_json(args.input.read_text(encoding="utf-8"))
    agent = AudioAgent()
    updated = agent.generate(state, output_dir=args.output_dir)

    payload = json.dumps(updated.model_dump(mode="json"), indent=2)
    output_path = args.output
    if output_path is None:
        output_path = args.output_dir / updated.meta.project_id / "state.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(f"Saved to {output_path}")

    if args.stdout:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
