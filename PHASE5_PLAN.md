# Phase 5 Implementation Plan: Intelligent Edit & Undo System (COMPLETED)

**STATUS: DONE** - The Phase 5 HITL conversational edit system has been fully implemented and verified!

This document outlines the architecture, breakdown, and implementation plan for Phase 5 of the Agentic AI Project. It incorporates LangGraph, Human-In-The-Loop (HITL), Model Context Protocol (MCP) servers, and a robust State Versioning/Undo system.

## High-Level Architecture

The goal of Phase 5 is to allow the user to describe edits in natural language (e.g., "Change the background in scene 2 to a desert"). The system must classify the intent, update the JSON state, re-run only the necessary pipeline segments (audio, video_frame, video, script), and save a snapshot of the state and assets. The user can revert to any previous snapshot.

We will use **LangGraph** to manage the conversational and execution flow, utilizing its built-in Checkpointer (`SqliteSaver`) for multi-turn memory.

---

## Module Breakdown

### Module 1: State Versioning & Undo System (High Priority)
**Location:** `state_manager/`
**Objective:** Enable full revert of the project at any granularity (assets + JSON state).

* **Components to Implement:**
  * `StateManager` class with a SQLite or append-only file backend.
  * `snapshot(version: str, state_json: dict, asset_paths: list) -> None`: Copies/hardlinks assets and saves the JSON state associated with the new version.
  * `revert(version: str) -> dict`: Restores the requested JSON state and resets the working asset directory to match that snapshot.
  * `history() -> list`: Returns a list of all versions with a diff summary.
* **Deliverable:** A functional `StateManager` capable of snapshotting and reverting the `data/outputs/` directory.

### Module 2: Intent Classification Agent
**Location:** `agents/edit_agent/intent_classifier.py`
**Objective:** Parse raw user text into structured, actionable JSON using LangChain/Groq.

* **Components to Implement:**
  * Pydantic Schema defining the output structure:
    ```json
    {
      "intent": "change_voice_tone",
      "target": "audio", // Enum: audio, video_frame, video, script
      "scope": "character:Narrator",
      "parameters": {"tone": "whispered"}
    }
    ```
  * LLM prompt and parsing logic to accurately categorize queries.
* **Deliverable:** The classifier script and a test suite covering at least 10 different edit query types (e.g., "Speed up this scene", "Change character design").

### Module 3: MCP Tools & Execution Logic
**Location:** `mcp/server.py` & `agents/edit_agent/executor.py`
**Objective:** Provide standard, callable tools for the agent to execute the classified intent.

* **Components to Implement:**
  * FastMCP Server exposing functions that map to the target types:
    * `regenerate_script(...)` (Target: script) -> Re-invokes Phase 1 LLM.
    * `regenerate_video_frames(...)` (Target: video_frame) -> Re-runs Image Generation (Phase 2).
    * `regenerate_audio(...)` (Target: audio) -> Re-runs TTS/Wav2Lip (Phase 3).
    * `recompose_video(...)` (Target: video) -> Re-runs FFmpeg compositing only.
* **Deliverable:** An MCP server providing clear abstraction over the underlying pipeline phases.

### Module 4: LangGraph Orchestrator & HITL
**Location:** `agents/orchestrator/graph.py`
**Objective:** Tie the intent classification, planning, and execution together into a stateful, interruptible workflow.

* **Components to Implement:**
  * `StateGraph` setup with `SqliteSaver` checkpointer for conversational memory.
  * **Nodes:**
    1. `Classify_Intent` -> Calls the Intent Classifier.
    2. `Plan_Edit` -> Updates the `state.json` based on intent parameters.
    3. `Human_Approval` -> **(HITL Interrupt)** Pauses execution. Prompts the user: "I plan to update Scene 2's background and re-run Image Generation. Proceed? (y/n)".
    4. `Execute_Edit` -> Calls the MCP tools to run the pipeline subset.
    5. `Snapshot_State` -> Calls `StateManager.snapshot`.
* **Deliverable:** The core orchestration graph that can be run interactively.

### Module 5: Interface (Low Priority)
**Location:** `run_phase5.py` / React UI (Later)
**Objective:** Provide a way for the user to interact with the edit graph and the undo system.

* **Components to Implement:**
  * CLI interactive loop connecting to the LangGraph application.
  * Commands to type natural language edits, or type `/undo v2` to trigger the `StateManager.revert`.
* **Deliverable:** A terminal application for testing and demonstrating the editing capabilities.

---

## Execution Order
1. ~~Setup dependencies (`langgraph`, `langchain-groq`, `mcp`).~~ (DONE)
2. ~~Build **Module 1** (State Manager) first, as every edit depends on saving state.~~ (DONE)
3. ~~Build **Module 2** (Intent Classifier) and write the 10 test cases.~~ (DONE)
4. ~~Build **Module 3** (Execution Logic) to allow partial pipeline runs.~~ (DONE - Implemented targeted asset deletion and universal state sync).
5. ~~Build **Module 4 & 5** (LangGraph & Interface) to tie everything together.~~ (DONE - Implemented conversational wrapper `get_project_context` and CLI `run_phase5.py`).

## Implementation Notes & Handoff Instructions
- **Conversational Awareness**: The LangGraph agent uses `get_project_context` to actively resolve ambiguity before overwriting traits.
- **Granular Reruns**: Editing characters or BGM triggers a selective deletion of matching cached assets (e.g., `scene_1_char_1.png`). When Phase 2/3 restarts, it automatically skips existing unmodified files and re-renders only what is missing.
- **ElevenLabs Free Tier Issue**: Free tier ElevenLabs accounts trigger a `401 Unauthorized` if multiple accounts/requests originate from proxies or VPNs. A fallback bypass using `TTS_PROVIDER=none` can be used to generate synthetic tones offline instead of crashing.
