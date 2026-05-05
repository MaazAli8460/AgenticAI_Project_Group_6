from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agents.audio_agent.agent import AudioAgent
from agents.edit_agent.executor import get_latest_state_path
from agents.orchestrator.graph import build_edit_graph
from agents.story_agent.agent import build_story_state
from agents.video_agent.agent import VideoAgent
from langchain_core.messages import AIMessage, HumanMessage
from state_manager.state_manager import StateManager
from shared.schemas.state import PipelineState

load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = APP_ROOT / "data" / "outputs"
PHASE1_DIR = OUTPUT_ROOT / "phase1"
PHASE2_DIR = OUTPUT_ROOT / "phase2"
PHASE3_DIR = OUTPUT_ROOT / "phase3"
PHASE3_LIP_DIR = OUTPUT_ROOT / "phase3_lip_sync"


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _lip_sync_enabled() -> bool:
	return os.getenv("LIP_SYNC_ENABLED", "0").lower() not in {"0", "false", "no"}


def _phase3_output_dir() -> Path:
	return PHASE3_LIP_DIR if _lip_sync_enabled() else PHASE3_DIR


@dataclass
class JobState:
	project_id: str
	prompt: Optional[str] = None
	status: str = "idle"
	message: str = ""
	phases: dict[str, str] = field(default_factory=lambda: {
		"phase1": "pending",
		"phase2": "pending",
		"phase3": "pending",
	})
	state_path: Optional[Path] = None
	video_path: Optional[Path] = None
	updated_at: str = field(default_factory=_now_iso)
	error: Optional[str] = None


JOB_STORE: dict[str, JobState] = {}
QUEUE_STORE: dict[str, asyncio.Queue[dict[str, Any]]] = {}
EDIT_GRAPH = None


def _get_queue(project_id: str) -> asyncio.Queue[dict[str, Any]]:
	if project_id not in QUEUE_STORE:
		QUEUE_STORE[project_id] = asyncio.Queue()
	return QUEUE_STORE[project_id]


def _emit(project_id: str, payload: dict[str, Any]) -> None:
	job = JOB_STORE.get(project_id)
	if job:
		job.updated_at = _now_iso()
		if "status" in payload:
			job.status = str(payload["status"])
		if "message" in payload:
			job.message = str(payload["message"])
		if "phase" in payload and "phase_status" in payload:
			job.phases[payload["phase"]] = str(payload["phase_status"])
		if "state_path" in payload:
			job.state_path = Path(payload["state_path"]) if payload["state_path"] else None
		if "video_path" in payload:
			job.video_path = Path(payload["video_path"]) if payload["video_path"] else None
		if "error" in payload:
			job.error = str(payload["error"])
	queue = _get_queue(project_id)
	queue.put_nowait(payload)


def _serialize_job(job: JobState) -> dict[str, Any]:
	return {
		"project_id": job.project_id,
		"prompt": job.prompt,
		"status": job.status,
		"message": job.message,
		"phases": job.phases,
		"state_path": str(job.state_path) if job.state_path else None,
		"video_path": str(job.video_path) if job.video_path else None,
		"updated_at": job.updated_at,
		"error": job.error,
	}


def _get_edit_graph():
	global EDIT_GRAPH
	if EDIT_GRAPH is None:
		EDIT_GRAPH = build_edit_graph()
	return EDIT_GRAPH


def _find_state_path(project_id: str) -> Optional[Path]:
	phase3_state = PHASE3_DIR / project_id / "state.json"
	if phase3_state.exists():
		return phase3_state
	lip_state = PHASE3_LIP_DIR / project_id / "state.json"
	if lip_state.exists():
		return lip_state
	phase2_state = PHASE2_DIR / project_id / "state.json"
	if phase2_state.exists():
		return phase2_state
	phase1_state = PHASE1_DIR / f"{project_id}.json"
	if phase1_state.exists():
		return phase1_state
	return None


def _find_video_path(project_id: str) -> Optional[Path]:
	state_path = _find_state_path(project_id)
	if state_path and state_path.exists():
		try:
			state = _load_pipeline_state(state_path)
			if state.video and state.video.final_video_file:
				candidate = Path(state.video.final_video_file)
				if candidate.exists():
					return candidate
		except Exception:
			pass
	for base_dir in (PHASE3_DIR, PHASE3_LIP_DIR):
		candidate = base_dir / project_id / "final" / f"{project_id}_final.mp4"
		if candidate.exists():
			return candidate
		candidate = base_dir / project_id / "final" / f"{project_id}_subtitled.mp4"
		if candidate.exists():
			return candidate
	return None


def _load_pipeline_state(path: Path) -> PipelineState:
	return PipelineState.model_validate_json(path.read_text(encoding="utf-8"))


class Phase1Request(BaseModel):
	prompt: str = Field(min_length=1)
	project_id: Optional[str] = None
	seed: Optional[int] = None


class PhaseRunRequest(BaseModel):
	project_id: str = Field(min_length=1)
	input_path: Optional[str] = None


class FullRunRequest(BaseModel):
	prompt: str = Field(min_length=1)
	project_id: Optional[str] = None
	seed: Optional[int] = None


class EditRequest(BaseModel):
	project_id: str = Field(min_length=1)
	message: str = Field(min_length=1)


class UndoRequest(BaseModel):
	project_id: str = Field(min_length=1)
	version: str = Field(min_length=1)


class UndoLatestRequest(BaseModel):
	project_id: str = Field(min_length=1)


app = FastAPI(title="Agentic AI Phase 4 API")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
	CORSMiddleware,
	allow_origins=[frontend_origin, "http://127.0.0.1:5173"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
	project_ids: set[str] = set()
	if PHASE1_DIR.exists():
		for path in PHASE1_DIR.glob("*.json"):
			project_ids.add(path.stem)
	if PHASE2_DIR.exists():
		for path in PHASE2_DIR.glob("*/state.json"):
			project_ids.add(path.parent.name)
	if PHASE3_DIR.exists():
		for path in PHASE3_DIR.glob("*/state.json"):
			project_ids.add(path.parent.name)
	if PHASE3_LIP_DIR.exists():
		for path in PHASE3_LIP_DIR.glob("*/state.json"):
			project_ids.add(path.parent.name)

	projects = []
	for project_id in sorted(project_ids):
		projects.append(
			{
				"project_id": project_id,
				"state_path": str(_find_state_path(project_id) or ""),
				"video_path": str(_find_video_path(project_id) or ""),
			}
		)
	return {"projects": projects}


@app.get("/api/status/{project_id}")
def get_status(project_id: str) -> dict[str, Any]:
	job = JOB_STORE.get(project_id)
	if job:
		return _serialize_job(job)
	state_path = _find_state_path(project_id)
	if not state_path:
		raise HTTPException(status_code=404, detail="Project not found")
	state = _load_pipeline_state(state_path)
	phases = {
		"phase1": "complete" if state.story and state.scenes else "pending",
		"phase2": "complete" if state.audio and state.audio.final_audio_file else "pending",
		"phase3": "complete" if state.video and state.video.final_video_file else "pending",
	}
	return {
		"project_id": project_id,
		"status": "complete",
		"message": "Loaded from disk",
		"phases": phases,
		"state_path": str(state_path),
		"video_path": str(_find_video_path(project_id) or ""),
		"updated_at": _now_iso(),
		"error": None,
	}


@app.get("/api/state/{project_id}")
def get_state(project_id: str) -> JSONResponse:
	state_path = _find_state_path(project_id)
	if not state_path:
		raise HTTPException(status_code=404, detail="State not found")
	return JSONResponse(content=json.loads(state_path.read_text(encoding="utf-8")))


@app.get("/api/video/{project_id}")
def get_video(project_id: str) -> FileResponse:
	video_path = _find_video_path(project_id)
	if not video_path:
		raise HTTPException(status_code=404, detail="Video not found")
	return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


@app.get("/api/stream/{project_id}")
async def stream_status(project_id: str) -> StreamingResponse:
	queue = _get_queue(project_id)

	async def event_stream():
		if project_id in JOB_STORE:
			yield f"data: {json.dumps(_serialize_job(JOB_STORE[project_id]))}\n\n"
		while True:
			try:
				payload = await asyncio.wait_for(queue.get(), timeout=15)
				yield f"data: {json.dumps(payload)}\n\n"
			except asyncio.TimeoutError:
				yield "event: ping\ndata: {}\n\n"

	response = StreamingResponse(event_stream(), media_type="text/event-stream")
	response.headers["Cache-Control"] = "no-cache"
	return response


def _ensure_job(project_id: str, prompt: Optional[str]) -> JobState:
	if project_id not in JOB_STORE:
		JOB_STORE[project_id] = JobState(project_id=project_id, prompt=prompt)
	else:
		if prompt:
			JOB_STORE[project_id].prompt = prompt
	return JOB_STORE[project_id]


def _run_phase1(prompt: str, project_id: str, seed: Optional[int]) -> JobState:
	_emit(project_id, {"message": "Starting Phase 1", "status": "running"})
	state = build_story_state(prompt, project_id=project_id, seed=seed)
	output_path = PHASE1_DIR / f"{state.meta.project_id}.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8"
	)
	job = _ensure_job(state.meta.project_id, prompt)
	_emit(
		state.meta.project_id,
		{
			"phase": "phase1",
			"phase_status": "complete",
			"message": "Phase 1 complete",
			"status": "running",
			"state_path": str(output_path),
		},
	)
	job.state_path = output_path
	return job


def _run_phase2(project_id: str, input_path: Optional[str]) -> JobState:
	job = _ensure_job(project_id, None)
	_emit(
		project_id,
		{
			"phase": "phase2",
			"phase_status": "running",
			"message": "Starting Phase 2",
			"status": "running",
		},
	)
	input_state = Path(input_path) if input_path else PHASE1_DIR / f"{project_id}.json"
	if not input_state.exists():
		raise RuntimeError(f"Phase 1 state not found: {input_state}")
	state = _load_pipeline_state(input_state)
	updated = AudioAgent().generate(state, output_dir=PHASE2_DIR)
	output_path = PHASE2_DIR / project_id / "state.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(updated.model_dump(mode="json"), indent=2), encoding="utf-8"
	)
	_emit(
		project_id,
		{
			"phase": "phase2",
			"phase_status": "complete",
			"message": "Phase 2 complete",
			"status": "running",
			"state_path": str(output_path),
		},
	)
	job.state_path = output_path
	return job


def _run_phase3(project_id: str, input_path: Optional[str]) -> JobState:
	job = _ensure_job(project_id, None)
	_emit(
		project_id,
		{
			"phase": "phase3",
			"phase_status": "running",
			"message": "Starting Phase 3",
			"status": "running",
		},
	)
	input_state = Path(input_path) if input_path else PHASE2_DIR / project_id / "state.json"
	if not input_state.exists():
		raise RuntimeError(f"Phase 2 state not found: {input_state}")
	state = _load_pipeline_state(input_state)
	output_dir = _phase3_output_dir()
	updated = VideoAgent().generate(state, output_dir=output_dir)
	output_path = output_dir / project_id / "state.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(updated.model_dump(mode="json"), indent=2), encoding="utf-8"
	)
	video_path = _find_video_path(project_id)
	_emit(
		project_id,
		{
			"phase": "phase3",
			"phase_status": "complete",
			"message": "Phase 3 complete",
			"status": "complete",
			"state_path": str(output_path),
			"video_path": str(video_path) if video_path else None,
		},
	)
	job.state_path = output_path
	job.video_path = video_path
	job.status = "complete"
	return job


def _apply_state_to_outputs(project_id: str, state_json: dict[str, Any]) -> None:
	paths = [
		PHASE3_LIP_DIR / project_id / "state.json",
		PHASE3_DIR / project_id / "state.json",
		PHASE2_DIR / project_id / "state.json",
		PHASE1_DIR / f"{project_id}.json",
	]
	wrote = False
	for path in paths:
		if path.exists():
			path.write_text(json.dumps(state_json, indent=2), encoding="utf-8")
			wrote = True
	if not wrote:
		PHASE1_DIR.mkdir(parents=True, exist_ok=True)
		target = PHASE1_DIR / f"{project_id}.json"
		target.write_text(json.dumps(state_json, indent=2), encoding="utf-8")


def _snapshot_edit(project_id: str, summary: str) -> Optional[str]:
	try:
		state_path = get_latest_state_path(project_id)
		state_json = json.loads(state_path.read_text(encoding="utf-8"))
		manager = StateManager(project_id=project_id, base_dir=OUTPUT_ROOT)
		version = f"v{len(manager.history()) + 1}"
		asset_paths = [
			PHASE1_DIR / f"{project_id}.json",
			PHASE2_DIR / project_id,
			PHASE3_DIR / project_id,
			PHASE3_LIP_DIR / project_id,
		]
		manager.snapshot(version, state_json, asset_paths, diff_summary=summary)
		return version
	except Exception:
		return None


def _run_edit_command(project_id: str, message: str) -> dict[str, Any]:
	graph = _get_edit_graph()
	config = {"configurable": {"thread_id": project_id}}
	tool_calls: list[str] = []
	agent_reply = ""
	tool_ran = False

	try:
		for event in graph.stream(
			{"project_id": project_id, "messages": [HumanMessage(content=message)]},
			config,
		):
			for node_name, node_output in event.items():
				if node_name == "agent":
					last_msg = node_output["messages"][-1]
					if isinstance(last_msg, AIMessage):
						agent_reply = last_msg.content
						if getattr(last_msg, "tool_calls", None):
							tool_calls = [call["name"] for call in last_msg.tool_calls]
				elif node_name == "tools":
					tool_ran = True
	
		snapshot_version = _snapshot_edit(project_id, message) if tool_ran else None
		return {
			"project_id": project_id,
			"reply": agent_reply,
			"tool_calls": tool_calls,
			"snapshot_version": snapshot_version,
		}
	except Exception as exc:
		return {
			"project_id": project_id,
			"reply": f"Edit agent failed: {exc}",
			"tool_calls": [],
			"snapshot_version": None,
			"error": str(exc),
		}


async def _run_in_thread(func, *args):
	return await asyncio.to_thread(func, *args)


@app.post("/api/run/phase1")
async def run_phase1(request: Phase1Request, background: BackgroundTasks) -> dict[str, Any]:
	project_id = request.project_id or f"proj_{uuid4().hex[:8]}"
	job = _ensure_job(project_id, request.prompt)
	job.status = "running"
	job.phases["phase1"] = "running"
	_emit(job.project_id, {"status": "running", "message": "Queued Phase 1"})

	async def _task() -> None:
		try:
			completed = await _run_in_thread(_run_phase1, request.prompt, project_id, request.seed)
			_emit(
				completed.project_id,
				{
					"status": "running",
					"message": "Phase 1 finished",
					"phase": "phase1",
					"phase_status": "complete",
					"state_path": str(completed.state_path) if completed.state_path else None,
				},
			)
		except Exception as exc:
			_emit(job.project_id, {"status": "error", "error": str(exc), "message": "Phase 1 failed"})

	background.add_task(_task)
	return {"status": "queued", "project_id": job.project_id}


@app.post("/api/run/phase2")
async def run_phase2(request: PhaseRunRequest, background: BackgroundTasks) -> dict[str, Any]:
	job = _ensure_job(request.project_id, None)
	job.status = "running"
	job.phases["phase2"] = "running"
	_emit(job.project_id, {"status": "running", "message": "Queued Phase 2"})

	async def _task() -> None:
		try:
			completed = await _run_in_thread(_run_phase2, request.project_id, request.input_path)
			_emit(
				completed.project_id,
				{
					"status": "running",
					"message": "Phase 2 finished",
					"phase": "phase2",
					"phase_status": "complete",
					"state_path": str(completed.state_path) if completed.state_path else None,
				},
			)
		except Exception as exc:
			_emit(job.project_id, {"status": "error", "error": str(exc), "message": "Phase 2 failed"})

	background.add_task(_task)
	return {"status": "queued", "project_id": job.project_id}


@app.post("/api/run/phase3")
async def run_phase3(request: PhaseRunRequest, background: BackgroundTasks) -> dict[str, Any]:
	job = _ensure_job(request.project_id, None)
	job.status = "running"
	job.phases["phase3"] = "running"
	_emit(job.project_id, {"status": "running", "message": "Queued Phase 3"})

	async def _task() -> None:
		try:
			completed = await _run_in_thread(_run_phase3, request.project_id, request.input_path)
			_emit(
				completed.project_id,
				{
					"status": "complete",
					"message": "Phase 3 finished",
					"phase": "phase3",
					"phase_status": "complete",
					"state_path": str(completed.state_path) if completed.state_path else None,
					"video_path": str(completed.video_path) if completed.video_path else None,
				},
			)
		except Exception as exc:
			_emit(job.project_id, {"status": "error", "error": str(exc), "message": "Phase 3 failed"})

	background.add_task(_task)
	return {"status": "queued", "project_id": job.project_id}


@app.post("/api/run/full")
async def run_full(request: FullRunRequest, background: BackgroundTasks) -> dict[str, Any]:
	project_id = request.project_id or f"proj_{uuid4().hex[:8]}"
	job = _ensure_job(project_id, request.prompt)
	job.status = "running"
	_emit(job.project_id, {"status": "running", "message": "Queued full pipeline"})

	async def _task() -> None:
		try:
			phase1_job = await _run_in_thread(_run_phase1, request.prompt, project_id, request.seed)
			await _run_in_thread(_run_phase2, phase1_job.project_id, None)
			await _run_in_thread(_run_phase3, phase1_job.project_id, None)
			_emit(
				phase1_job.project_id,
				{
					"status": "complete",
					"message": "Pipeline complete",
					"video_path": str(_find_video_path(phase1_job.project_id) or ""),
				},
			)
		except Exception as exc:
			_emit(job.project_id, {"status": "error", "error": str(exc), "message": "Pipeline failed"})

	background.add_task(_task)
	return {"status": "queued", "project_id": job.project_id}


@app.post("/api/edit")
async def edit_project(request: EditRequest) -> dict[str, Any]:
	job = _ensure_job(request.project_id, None)
	_emit(job.project_id, {"status": "running", "message": "Edit agent running"})

	result = await _run_in_thread(_run_edit_command, request.project_id, request.message)
	_emit(
		job.project_id,
		{
			"status": "complete",
			"message": "Edit agent completed",
			"phase": "edit",
			"phase_status": "complete",
		},
	)
	return result


@app.get("/api/edit/history/{project_id}")
def edit_history(project_id: str) -> dict[str, Any]:
	manager = StateManager(project_id=project_id, base_dir=OUTPUT_ROOT)
	return {"history": manager.history()}


@app.post("/api/edit/undo")
async def edit_undo(request: UndoRequest) -> dict[str, Any]:
	job = _ensure_job(request.project_id, None)
	_emit(job.project_id, {"status": "running", "message": f"Undo {request.version}"})

	def _do_revert() -> dict[str, Any]:
		manager = StateManager(project_id=request.project_id, base_dir=OUTPUT_ROOT)
		state_json = manager.revert(request.version)
		_apply_state_to_outputs(request.project_id, state_json)
		return {"status": "reverted", "version": request.version}

	result = await _run_in_thread(_do_revert)
	_emit(job.project_id, {"status": "complete", "message": "Undo complete"})
	return result


@app.post("/api/edit/undo_latest")
async def edit_undo_latest(request: UndoLatestRequest) -> dict[str, Any]:
	job = _ensure_job(request.project_id, None)
	manager = StateManager(project_id=request.project_id, base_dir=OUTPUT_ROOT)
	previous_version = manager.previous_version()
	if not previous_version:
		return {"status": "noop", "message": "No previous snapshot available."}
	_emit(job.project_id, {"status": "running", "message": f"Undo {previous_version}"})

	def _do_revert() -> dict[str, Any]:
		state_json = manager.revert(previous_version)
		_apply_state_to_outputs(request.project_id, state_json)
		return {"status": "reverted", "version": previous_version}

	result = await _run_in_thread(_do_revert)
	_emit(job.project_id, {"status": "complete", "message": "Undo complete"})
	return result
