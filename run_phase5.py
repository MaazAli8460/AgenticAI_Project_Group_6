import argparse
import sys
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agents.orchestrator.graph import build_edit_graph
from state_manager.state_manager import StateManager

def interactive_loop(project_id: str):
    print(f"\n[START] Starting Edit Agent for Project: {project_id}")
    print("Type your edit request (e.g. 'Make scene 2 look like a cyberpunk city').")
    print("Commands:")
    print("  /undo <version>  - Revert the project to a specific snapshot (e.g. /undo v1)")
    print("  /history         - View snapshot history")
    print("  exit             - Quit the editor\n")
    
    graph = build_edit_graph()
    
    # Each editing session for a project gets its own thread for memory
    config = {"configurable": {"thread_id": project_id}}
    
    # Initialize the state manager for this project
    state_manager = StateManager(project_id=project_id, base_dir="data/outputs")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if not user_input:
            continue
            
        if user_input.startswith("/history"):
            history = state_manager.history()
            if not history:
                print("No history available.")
            else:
                for h in history:
                    print(f"- {h['version']} [{h['timestamp']}]: {h['diff_summary']}")
            continue
            
        if user_input.startswith("/undo "):
            version = user_input.split(" ")[1].strip()
            print(f"Reverting to {version}...")
            try:
                state_manager.revert(version)
                print(f"[SUCCESS] Successfully reverted to {version}.")
            except Exception as e:
                print(f"[ERROR] Revert failed: {e}")
            continue

        # Standard natural language request
        msg = HumanMessage(content=user_input)
        
        # We start tracking snapshot versions
        current_version_count = len(state_manager.history()) + 1
        
        print("\n[Agent is thinking...]")
        
        # Run the graph
        # Since we use interrupts or wait states, we can stream the output
        for event in graph.stream({"project_id": project_id, "messages": [msg]}, config):
            for node_name, node_output in event.items():
                if node_name == "agent":
                    agent_message = node_output["messages"][-1]
                    print(f"\n[Agent]: {agent_message.content}")
                    
                    if hasattr(agent_message, "tool_calls") and agent_message.tool_calls:
                        print(f"   [Calling tools: {[t['name'] for t in agent_message.tool_calls]}]")
                        
                elif node_name == "tools":
                    # Tool executed. Let's take a snapshot if a modification happened.
                    # This could be improved to only snapshot on successful execution, 
                    # but for now, we snapshot after tools are run.
                    print("\n📸 Auto-saving state snapshot...")
                    latest_state = {} # Load state logic here if needed
                    try:
                        from agents.edit_agent.executor import get_latest_state_path
                        import json
                        with open(get_latest_state_path(project_id), "r") as f:
                            latest_state = json.load(f)
                        
                        # Specify paths to backup
                        from pathlib import Path
                        asset_paths = [
                            Path("data/outputs/phase1") / f"{project_id}.json",
                            Path("data/outputs/phase2") / project_id,
                            Path("data/outputs/phase3") / project_id,
                            Path("data/outputs/phase3_lip_sync") / project_id,
                        ]
                        state_manager.snapshot(f"v{current_version_count}", latest_state, asset_paths, diff_summary=user_input)
                        print(f"[SUCCESS] Snapshot v{current_version_count} saved. You can revert via '/undo v{current_version_count}'.")
                    except Exception as e:
                        print(f"Failed to auto-snapshot: {e}")
                
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 5 interactive Edit Agent.")
    parser.add_argument("--project-id", required=True, help="The project ID to edit (e.g. proj_abcd1234)")
    args = parser.parse_args()
    
    interactive_loop(args.project_id)
