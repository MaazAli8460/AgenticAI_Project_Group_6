import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const phaseLabels = {
  phase1: "Story",
  phase2: "Audio",
  phase3: "Video",
};

const emptyPhases = {
  phase1: "pending",
  phase2: "pending",
  phase3: "pending",
};

function statusTone(status) {
  if (status === "complete") return "status ok";
  if (status === "running") return "status running";
  if (status === "error") return "status error";
  return "status idle";
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [projectId, setProjectId] = useState("");
  const [status, setStatus] = useState({
    status: "idle",
    message: "",
    phases: { ...emptyPhases },
    updated_at: "",
    video_path: "",
  });
  const [events, setEvents] = useState([]);
  const [projects, setProjects] = useState([]);
  const [isBusy, setIsBusy] = useState(false);
  const [editMessage, setEditMessage] = useState("");
  const [editReply, setEditReply] = useState("");
  const [editHistory, setEditHistory] = useState([]);
  const [editBusy, setEditBusy] = useState(false);
  const [editThread, setEditThread] = useState([]);

  const videoUrl = useMemo(() => {
    if (!projectId) return "";
    const stamp = status.updated_at ? encodeURIComponent(status.updated_at) : "";
    return `${API_BASE}/api/video/${projectId}${stamp ? `?t=${stamp}` : ""}`;
  }, [projectId, status.updated_at]);

  const stateUrl = useMemo(() => {
    if (!projectId) return "";
    return `${API_BASE}/api/state/${projectId}`;
  }, [projectId]);

  useEffect(() => {
    fetchProjects();
  }, []);

  useEffect(() => {
    if (!projectId) return;
    const stream = new EventSource(`${API_BASE}/api/stream/${projectId}`);

    stream.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleEvent(payload);
      } catch (error) {
        console.error("Stream parse error", error);
      }
    };

    stream.onerror = () => {
      stream.close();
    };

    return () => stream.close();
  }, [projectId]);

  useEffect(() => {
    if (!projectId) return;
    fetchHistory(projectId);
  }, [projectId]);

  function handleEvent(payload) {
    setEvents((prev) => [payload, ...prev].slice(0, 12));
    setStatus((prev) => {
      const next = { ...prev };
      if (payload.status) next.status = payload.status;
      if (payload.message) next.message = payload.message;
      if (payload.updated_at) next.updated_at = payload.updated_at;
      if (payload.video_path) next.video_path = payload.video_path;
      if (payload.phases) next.phases = payload.phases;
      if (payload.phase && payload.phase_status) {
        next.phases = { ...next.phases, [payload.phase]: payload.phase_status };
      }
      return next;
    });
    if (payload.project_id && payload.project_id !== projectId) {
      setProjectId(payload.project_id);
    }
  }

  async function fetchProjects() {
    try {
      const response = await fetch(`${API_BASE}/api/projects`);
      const data = await response.json();
      setProjects(data.projects || []);
    } catch (error) {
      console.error(error);
    }
  }

  async function fetchStatus(targetId) {
    try {
      const response = await fetch(`${API_BASE}/api/status/${targetId}`);
      const data = await response.json();
      handleEvent(data);
    } catch (error) {
      console.error(error);
    }
  }

  async function fetchHistory(targetId) {
    try {
      const response = await fetch(`${API_BASE}/api/edit/history/${targetId}`);
      const data = await response.json();
      setEditHistory(data.history || []);
    } catch (error) {
      console.error(error);
    }
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Request failed");
    }
    return response.json();
  }

  async function runFull() {
    if (!prompt.trim()) {
      alert("Please enter a prompt.");
      return;
    }
    setIsBusy(true);
    try {
      const result = await postJson(`${API_BASE}/api/run/full`, { prompt });
      setProjectId(result.project_id);
      setStatus((prev) => ({ ...prev, status: "running" }));
    } catch (error) {
      alert(error.message || "Failed to run pipeline");
    } finally {
      setIsBusy(false);
      fetchProjects();
    }
  }

  async function runPhase(phase) {
    if (phase === "phase1") {
      if (!prompt.trim()) {
        alert("Please enter a prompt.");
        return;
      }
      setIsBusy(true);
      try {
        const result = await postJson(`${API_BASE}/api/run/phase1`, { prompt, project_id: projectId || undefined });
        setProjectId(result.project_id);
      } catch (error) {
        alert(error.message || "Phase 1 failed");
      } finally {
        setIsBusy(false);
        fetchProjects();
      }
      return;
    }

    if (!projectId) {
      alert("Set a project ID first.");
      return;
    }

    setIsBusy(true);
    try {
      await postJson(`${API_BASE}/api/run/${phase}`, { project_id: projectId });
      setStatus((prev) => ({ ...prev, status: "running" }));
    } catch (error) {
      alert(error.message || "Phase run failed");
    } finally {
      setIsBusy(false);
      fetchProjects();
    }
  }

  async function sendEdit() {
    if (!projectId) {
      alert("Set a project ID first.");
      return;
    }
    const outgoing = editMessage.trim();
    if (!outgoing) {
      alert("Enter an edit request.");
      return;
    }
    setEditBusy(true);
    setEditThread((prev) => [
      { role: "user", text: outgoing, time: new Date().toLocaleTimeString() },
      ...prev,
    ].slice(0, 8));
    try {
      const result = await postJson(`${API_BASE}/api/edit`, {
        project_id: projectId,
        message: outgoing,
      });
      setEditReply(result.reply || "");
      if (result.reply) {
        setEditThread((prev) => [
          { role: "agent", text: result.reply, time: new Date().toLocaleTimeString() },
          ...prev,
        ].slice(0, 8));
      }
      setEvents((prev) => [
        {
          message: "Edit agent reply",
          status: "complete",
          phase: "edit",
          phase_status: "complete",
        },
        ...prev,
      ]);
      setEditMessage("");
      fetchHistory(projectId);
      fetchStatus(projectId);
    } catch (error) {
      alert(error.message || "Edit failed");
    } finally {
      setEditBusy(false);
    }
  }

  async function undoVersion(version) {
    if (!projectId) return;
    setEditBusy(true);
    try {
      await postJson(`${API_BASE}/api/edit/undo`, { project_id: projectId, version });
      fetchHistory(projectId);
      fetchStatus(projectId);
    } catch (error) {
      alert(error.message || "Undo failed");
    } finally {
      setEditBusy(false);
    }
  }

  async function undoLatest() {
    if (!projectId) return;
    setEditBusy(true);
    try {
      const result = await postJson(`${API_BASE}/api/edit/undo_latest`, { project_id: projectId });
      if (result.status === "noop") {
        alert(result.message || "No previous snapshot available.");
      }
      fetchHistory(projectId);
      fetchStatus(projectId);
    } catch (error) {
      alert(error.message || "Undo failed");
    } finally {
      setEditBusy(false);
    }
  }

  function loadProject(id) {
    setProjectId(id);
    fetchStatus(id);
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Agentic AI Studio</p>
          <h1>From prompt to polished short film.</h1>
          <p className="subhead">
            Orchestrate story, audio, and video generation with live progress and phase controls.
          </p>
        </div>
        <div className="hero-panel">
          <div className="field">
            <label>Prompt</label>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="A lone explorer follows a signal into a hidden storm..."
            />
          </div>
          <div className="row">
            <div className="field small">
              <label>Project ID</label>
              <input
                value={projectId}
                onChange={(event) => setProjectId(event.target.value)}
                placeholder="proj_..."
              />
            </div>
            <button className="primary" onClick={runFull} disabled={isBusy}>
              Run Full Pipeline
            </button>
          </div>
          <div className="actions">
            <button onClick={() => runPhase("phase1")} disabled={isBusy}>
              Run Phase 1
            </button>
            <button onClick={() => runPhase("phase2")} disabled={isBusy}>
              Run Phase 2
            </button>
            <button onClick={() => runPhase("phase3")} disabled={isBusy}>
              Run Phase 3
            </button>
            <button className="ghost" onClick={() => fetchStatus(projectId)} disabled={!projectId}>
              Refresh Status
            </button>
          </div>
        </div>
      </header>

      <main className="content">
        <section className="panel">
          <h2>Pipeline Status</h2>
          <div className="status-grid">
            {Object.entries(phaseLabels).map(([phase, label]) => (
              <div key={phase} className="status-card">
                <div>
                  <p className="label">{label}</p>
                  <p className="phase">{phase.toUpperCase()}</p>
                </div>
                <span className={statusTone(status.phases?.[phase] || "pending")}>
                  {status.phases?.[phase] || "pending"}
                </span>
              </div>
            ))}
          </div>
          <p className="status-message">{status.message || "Waiting for a run."}</p>
          {status.error && <p className="status-error">{status.error}</p>}
        </section>

        <section className="panel">
          <h2>Live Events</h2>
          <div className="events">
            {events.length === 0 && <p className="muted">No events yet.</p>}
            {events.map((event, index) => (
              <div key={index} className="event-item">
                <div>
                  <p className="event-title">{event.message || "Update"}</p>
                  <p className="muted">{event.phase ? `${event.phase}: ${event.phase_status}` : event.status}</p>
                </div>
                <span className={statusTone(event.status || "idle")}>{event.status || "update"}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <h2>Preview</h2>
          <div className="preview">
            {projectId ? (
              <>
                <video controls src={videoUrl} />
                <div className="preview-actions">
                  <a className="ghost" href={videoUrl} download>
                    Download MP4
                  </a>
                  <a className="ghost" href={stateUrl} target="_blank" rel="noreferrer">
                    View State JSON
                  </a>
                </div>
              </>
            ) : (
              <p className="muted">Run a project to see a preview.</p>
            )}
          </div>
        </section>

        <section className="panel">
          <h2>Edit Agent</h2>
          <div className="field">
            <label>Edit request</label>
            <textarea
              value={editMessage}
              onChange={(event) => setEditMessage(event.target.value)}
              placeholder="Make scene 2 feel like a cyberpunk market at night."
            />
          </div>
          <div className="actions">
            <button onClick={sendEdit} disabled={editBusy}>
              Send Edit
            </button>
            <button className="ghost" onClick={() => fetchHistory(projectId)} disabled={!projectId}>
              Refresh History
            </button>
          </div>
          <div className="edit-reply">
            <div className="edit-reply-header">
              <span>Latest reply</span>
              {editBusy && <span className="muted">Working...</span>}
            </div>
            {editReply ? <p>{editReply}</p> : <p className="muted">No edit response yet.</p>}
          </div>
          <div className="edit-thread">
            <div className="edit-reply-header">
              <span>Chat log</span>
              <span className="muted">Last 8</span>
            </div>
            {editThread.length === 0 && <p className="muted">No messages yet.</p>}
            {editThread.map((item, index) => (
              <div key={`${item.role}-${index}`} className={`edit-bubble ${item.role}`}>
                <div className="edit-meta">
                  <span className="edit-role">{item.role === "user" ? "You" : "Agent"}</span>
                  <span className="edit-time">{item.time}</span>
                </div>
                <p>{item.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <h2>Undo History</h2>
          <div className="actions">
            <button onClick={undoLatest} disabled={editBusy || !projectId}>
              Undo Latest
            </button>
          </div>
          <div className="history">
            {editHistory.length === 0 && <p className="muted">No snapshots yet.</p>}
            {editHistory.map((item) => (
              <div key={item.version} className="history-item">
                <div>
                  <p className="label">{item.version}</p>
                  <p className="muted">{item.diff_summary || "Snapshot"}</p>
                </div>
                <button onClick={() => undoVersion(item.version)} disabled={editBusy}>
                  Undo
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <h2>Recent Projects</h2>
          <div className="projects">
            {projects.length === 0 && <p className="muted">No projects yet.</p>}
            {projects.map((item) => (
              <button key={item.project_id} onClick={() => loadProject(item.project_id)}>
                <div>
                  <p className="label">{item.project_id}</p>
                  <p className="muted">{item.video_path ? "Video ready" : "In progress"}</p>
                </div>
                <span className="chevron">-&gt;</span>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
