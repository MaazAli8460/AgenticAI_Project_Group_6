from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agents.story_agent.agent import build_story_state


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Phase 1 story generation")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="A traveler finds a hidden valley.",
        help="Prompt to generate the story from",
    )
    parser.add_argument("--project-id", dest="project_id")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the PipelineState JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/phase1"),
        help="Folder to store the Phase 1 JSON output",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout in addition to saving",
    )
    args = parser.parse_args()

    state = build_story_state(args.prompt, project_id=args.project_id, seed=args.seed)
    payload = json.dumps(state.model_dump(mode="json"), indent=2)

    output_path = args.output
    if output_path is None:
        output_path = args.output_dir / f"{state.meta.project_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(f"Saved to {output_path}")

    if args.stdout:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
