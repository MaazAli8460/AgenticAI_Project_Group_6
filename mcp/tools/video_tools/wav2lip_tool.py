from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class Wav2LipError(RuntimeError):
	"""Raised when a Wav2Lip inference call fails or produces no output."""


class Wav2LipTool:
	"""Subprocess wrapper around the Rudrabha/Wav2Lip ``inference.py`` script.

	Wav2Lip lives outside this repository because it ships its own
	dependencies (PyTorch, librosa, opencv) that often conflict with the
	project's main environment. Configure it via env vars:

	- ``WAV2LIP_DIR``: absolute path to the cloned Wav2Lip repository.
	- ``WAV2LIP_CHECKPOINT``: absolute path to ``wav2lip_gan.pth`` (or ``wav2lip.pth``).
	- ``WAV2LIP_PYTHON``: optional Python interpreter (e.g. a Wav2Lip-only venv).
	  Falls back to the current interpreter.
	- ``WAV2LIP_REMOTE_URL``: when set, inference is offloaded to a remote
	  server (e.g. Google Colab via ngrok). The local Wav2Lip repo and
	  checkpoint are not required in this mode.

	If the configured paths do not resolve, ``is_available()`` returns False and
	callers can fall back to non-lip-synced rendering.
	"""

	def __init__(
		self,
		repo_dir: Optional[str | Path] = None,
		checkpoint_path: Optional[str | Path] = None,
		python_executable: Optional[str] = None,
		pads: Optional[str] = None,
		resize_factor: Optional[int] = None,
		wav2lip_batch_size: Optional[int] = None,
		face_det_batch_size: Optional[int] = None,
		nosmooth: Optional[bool] = None,
		debug: Optional[bool] = None,
		gpu_id: Optional[int] = None,
		remote_url: Optional[str] = None,
	) -> None:
		# Remote inference URL (Colab/ngrok)
		self._remote_url = remote_url or os.getenv("WAV2LIP_REMOTE_URL") or None
		if self._remote_url:
			self._remote_url = self._remote_url.rstrip("/")

		repo_env = repo_dir if repo_dir is not None else os.getenv("WAV2LIP_DIR")
		ckpt_env = checkpoint_path if checkpoint_path is not None else os.getenv(
			"WAV2LIP_CHECKPOINT"
		)
		self._repo_dir: Optional[Path] = (
			Path(str(repo_env)).expanduser() if repo_env else None
		)
		self._checkpoint: Optional[Path] = (
			Path(str(ckpt_env)).expanduser() if ckpt_env else None
		)
		self._python = python_executable or os.getenv("WAV2LIP_PYTHON") or sys.executable
		self._pads = pads or os.getenv("WAV2LIP_PADS", "0,10,0,0")
		self._resize_factor = (
			resize_factor
			if resize_factor is not None
			else int(os.getenv("WAV2LIP_RESIZE_FACTOR", "1"))
		)
		self._wav2lip_batch_size = (
			wav2lip_batch_size
			if wav2lip_batch_size is not None
			else int(os.getenv("WAV2LIP_BATCH_SIZE", "8"))
		)
		self._face_det_batch_size = (
			face_det_batch_size
			if face_det_batch_size is not None
			else int(os.getenv("WAV2LIP_FACE_DET_BATCH_SIZE", "4"))
		)
		self._gpu_id = (
			gpu_id
			if gpu_id is not None
			else int(os.getenv("WAV2LIP_GPU_ID", "0"))
		)
		self._nosmooth = (
			nosmooth
			if nosmooth is not None
			else os.getenv("WAV2LIP_NOSMOOTH", "0").lower() in {"1", "true", "yes"}
		)
		self._debug = (
			debug
			if debug is not None
			else os.getenv("LIP_SYNC_DEBUG", "0").lower() in {"1", "true", "yes"}
		)

	def is_available(self) -> bool:
		return self.availability_reason() is None

	def availability_reason(self) -> Optional[str]:
		# Remote mode: only need a reachable URL
		if self._remote_url:
			return None
		if not self._repo_dir:
			return "WAV2LIP_DIR is not set"
		if not self._checkpoint:
			return "WAV2LIP_CHECKPOINT is not set"
		if not self._repo_dir.is_dir():
			return f"WAV2LIP_DIR does not exist: {self._repo_dir}"
		if not (self._repo_dir / "inference.py").exists():
			return f"inference.py not found in {self._repo_dir}"
		if not self._checkpoint.exists():
			return f"checkpoint not found: {self._checkpoint}"
		return None

	def lip_sync(
		self,
		face_path: Path,
		audio_path: Path,
		output_path: Path,
	) -> Path:
		reason = self.availability_reason()
		if reason is not None:
			raise Wav2LipError(reason)
		if not face_path.exists():
			raise Wav2LipError(f"face input missing: {face_path}")
		if not audio_path.exists():
			raise Wav2LipError(f"audio input missing: {audio_path}")

		output_path.parent.mkdir(parents=True, exist_ok=True)

		# ── Remote mode ──
		if self._remote_url:
			return self._lip_sync_remote(face_path, audio_path, output_path)

		# ── Local mode ──
		return self._lip_sync_local(face_path, audio_path, output_path)

	# ────────────────────────────────────────────────────────────
	#  Remote inference (Colab / ngrok)
	# ────────────────────────────────────────────────────────────
	def _lip_sync_remote(
		self,
		face_path: Path,
		audio_path: Path,
		output_path: Path,
	) -> Path:
		"""Send face video + audio to the remote Wav2Lip server and save the result."""
		import urllib.request
		import urllib.error
		import io

		url = f"{self._remote_url}/lip_sync"

		if self._debug:
			print(f"[wav2lip-remote] POST {url}", flush=True)
			print(f"[wav2lip-remote]   face  = {face_path}", flush=True)
			print(f"[wav2lip-remote]   audio = {audio_path}", flush=True)

		# Build multipart/form-data request manually (no external deps)
		boundary = "----Wav2LipBoundary9876543210"
		body = io.BytesIO()

		def _write_field(name: str, value: str) -> None:
			body.write(f"--{boundary}\r\n".encode())
			body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
			body.write(f"{value}\r\n".encode())

		def _write_file(field: str, filepath: Path) -> None:
			body.write(f"--{boundary}\r\n".encode())
			body.write(
				f'Content-Disposition: form-data; name="{field}"; '
				f'filename="{filepath.name}"\r\n'.encode()
			)
			body.write(b"Content-Type: application/octet-stream\r\n\r\n")
			body.write(filepath.read_bytes())
			body.write(b"\r\n")

		# Form fields
		_write_field("pads", self._pads.replace(" ", ""))
		_write_field("resize_factor", str(self._resize_factor))
		_write_field("wav2lip_batch_size", str(self._wav2lip_batch_size))
		_write_field("face_det_batch_size", str(self._face_det_batch_size))
		_write_field("nosmooth", "1" if self._nosmooth else "0")

		# Files
		_write_file("face", face_path)
		_write_file("audio", audio_path)

		body.write(f"--{boundary}--\r\n".encode())
		body_bytes = body.getvalue()

		req = urllib.request.Request(
			url,
			data=body_bytes,
			headers={
				"Content-Type": f"multipart/form-data; boundary={boundary}",
				"Content-Length": str(len(body_bytes)),
			},
			method="POST",
		)

		try:
			# Long timeout because inference can take minutes
			with urllib.request.urlopen(req, timeout=600) as resp:
				if resp.status != 200:
					raise Wav2LipError(
						f"Remote Wav2Lip returned HTTP {resp.status}: "
						f"{resp.read().decode(errors='ignore')}"
					)
				output_path.write_bytes(resp.read())
		except urllib.error.HTTPError as exc:
			error_body = exc.read().decode(errors="ignore") if exc.fp else ""
			raise Wav2LipError(
				f"Remote Wav2Lip failed (HTTP {exc.code}): {error_body}"
			) from exc
		except urllib.error.URLError as exc:
			raise Wav2LipError(
				f"Cannot reach remote Wav2Lip at {self._remote_url}: {exc.reason}"
			) from exc

		if not output_path.exists() or output_path.stat().st_size == 0:
			raise Wav2LipError("Remote Wav2Lip returned empty output.")

		if self._debug:
			size_kb = output_path.stat().st_size / 1024
			print(f"[wav2lip-remote] wrote {output_path} ({size_kb:.0f} KB)", flush=True)

		return output_path

	# ────────────────────────────────────────────────────────────
	#  Local inference (subprocess)
	# ────────────────────────────────────────────────────────────
	def _lip_sync_local(
		self,
		face_path: Path,
		audio_path: Path,
		output_path: Path,
	) -> Path:
		assert self._repo_dir is not None  # narrowed by availability check
		assert self._checkpoint is not None

		pads_list = [piece.strip() for piece in str(self._pads).split(",") if piece.strip()]
		if len(pads_list) != 4:
			pads_list = ["0", "10", "0", "0"]

		cmd: list[str] = [
			self._python,
			str((self._repo_dir / "inference.py").resolve()),
			"--checkpoint_path",
			str(self._checkpoint.resolve()),
			"--face",
			str(face_path.resolve()),
			"--audio",
			str(audio_path.resolve()),
			"--outfile",
			str(output_path.resolve()),
			"--pads",
			*pads_list,
			"--wav2lip_batch_size",
			str(self._wav2lip_batch_size),
			"--face_det_batch_size",
			str(self._face_det_batch_size),
			"--resize_factor",
			str(self._resize_factor),
			"--gpu_id",
			str(self._gpu_id),
		]
		if self._nosmooth:
			cmd.append("--nosmooth")

		if self._debug:
			print(f"[wav2lip] running: {' '.join(cmd)}", flush=True)

		try:
			result = subprocess.run(
				cmd,
				cwd=str(self._repo_dir),
				check=True,
				capture_output=True,
			)
		except FileNotFoundError as exc:
			raise Wav2LipError(
				f"Failed to launch Wav2Lip interpreter '{self._python}': {exc}"
			) from exc
		except subprocess.CalledProcessError as exc:
			stdout = exc.stdout.decode(errors="ignore") if exc.stdout else ""
			stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
			raise Wav2LipError(
				"Wav2Lip inference failed.\n"
				f"Command: {' '.join(cmd)}\n"
				f"Stdout: {stdout}\nStderr: {stderr}"
			) from exc

		if not output_path.exists() or output_path.stat().st_size == 0:
			stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
			stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
			raise Wav2LipError(
				"Wav2Lip produced no output.\n"
				f"Command: {' '.join(cmd)}\n"
				f"Stdout: {stdout}\nStderr: {stderr}"
			)

		if self._debug:
			print(f"[wav2lip] wrote: {output_path}", flush=True)

		return output_path
