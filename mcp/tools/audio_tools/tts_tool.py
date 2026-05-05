from __future__ import annotations

import asyncio
import audioop
import os
import subprocess
import sys
import wave
from array import array
from pathlib import Path
from typing import Optional

import httpx

from shared.schemas.state import VoiceProfile

from .audio_utils import (
	SAMPLE_RATE,
	estimate_duration_ms,
	generate_tone_samples,
	read_wav,
	write_wav,
)


STYLE_FREQUENCIES = {
	"calm": 220.0,
	"soft": 260.0,
	"bright": 330.0,
	"neutral": 240.0,
	"warm": 200.0,
}

STYLE_KEYWORDS = {
	"calm": ["calm", "soothing", "gentle", "soft"],
	"soft": ["soft", "warm", "intimate"],
	"bright": ["bright", "energetic", "lively", "cheerful"],
	"neutral": ["neutral", "clear", "balanced"],
	"warm": ["warm", "friendly", "rich"],
	"serious": ["serious", "authoritative", "firm"],
	"whispered": ["whisper", "breathy", "hushed"],
}


class TtsTool:
	def __init__(
		self,
		provider: Optional[str] = None,
		base_url: str = "https://api.elevenlabs.io/v1",
		timeout_s: float = 60.0,
	) -> None:
		provider_value = provider or os.getenv("TTS_PROVIDER", "auto")
		provider_slug = str(provider_value).strip().lower()
		if provider_slug in {"edge-tts", "edge_tts"}:
			provider_slug = "edge"
		self._provider = provider_slug
		self._debug = os.getenv("TTS_DEBUG", "").lower() in {"1", "true", "yes"}
		self._strict = os.getenv("TTS_STRICT", "").lower() in {"1", "true", "yes"}
		self._api_key = os.getenv("ELEVENLABS_API_KEY")
		self._voice_id_default = os.getenv("ELEVENLABS_VOICE_ID") or os.getenv(
			"ELEVENLABS_VOICE_ID_DEFAULT"
		)
		self._voice_id_by_gender = {
			"female": os.getenv("ELEVENLABS_VOICE_ID_FEMALE"),
			"male": os.getenv("ELEVENLABS_VOICE_ID_MALE"),
			"neutral": os.getenv("ELEVENLABS_VOICE_ID_NEUTRAL"),
		}
		self._model_id = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
		self._output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "wav_22050")
		self._client = (
			httpx.Client(base_url=base_url, timeout=timeout_s)
			if self._api_key
			else None
		)
		self._voice_catalog: Optional[list[dict[str, object]]] = None
		self._voice_cache: dict[str, str] = {}
		self._piper_bin = os.getenv("PIPER_BIN") or "piper"
		self._piper_model = os.getenv("PIPER_MODEL")
		self._piper_config = os.getenv("PIPER_CONFIG")
		self._piper_speaker = os.getenv("PIPER_SPEAKER")
		self._edge_voice_default = os.getenv("EDGE_TTS_VOICE") or os.getenv(
			"EDGE_TTS_VOICE_DEFAULT"
		)
		self._edge_voice_by_gender = {
			"female": os.getenv("EDGE_TTS_VOICE_FEMALE"),
			"male": os.getenv("EDGE_TTS_VOICE_MALE"),
			"neutral": os.getenv("EDGE_TTS_VOICE_NEUTRAL"),
		}
		self._edge_rate = os.getenv("EDGE_TTS_RATE", "+0%")
		self._edge_pitch = os.getenv("EDGE_TTS_PITCH", "+0Hz")
		self._edge_volume = os.getenv("EDGE_TTS_VOLUME", "+0%")
		self._edge_output_format = os.getenv(
			"EDGE_TTS_OUTPUT_FORMAT",
			"riff-24khz-16bit-mono-pcm",
		)

	def synthesize(self, text: str, voice: VoiceProfile, output_path: Path) -> int:
		if self._can_use_elevenlabs():
			voice_id = self._resolve_voice_id(voice)
			if voice_id:
				self._maybe_warn(f"Using ElevenLabs voice_id={voice_id}")
				try:
					return self._synthesize_elevenlabs(text, voice, output_path, voice_id)
				except Exception as exc:
					if "404" in str(exc):
						self._maybe_warn("Voice id not found; refreshing voice list and retrying.")
						self._refresh_voice_catalog()
						retry_id = self._resolve_voice_id(voice, exclude_ids={voice_id})
						if retry_id:
							self._maybe_warn(f"Retrying with voice_id={retry_id}")
							return self._synthesize_elevenlabs(
								text, voice, output_path, retry_id
							)
					self._maybe_warn(f"ElevenLabs TTS failed: {exc}")
					if self._provider == "elevenlabs" or self._strict:
						raise
			else:
				self._maybe_warn("ElevenLabs voice_id could not be resolved; using fallback.")
				if self._provider == "elevenlabs" or self._strict:
					raise RuntimeError("ElevenLabs voice_id resolution failed.")
		if self._can_use_edge():
			edge_voice = self._resolve_edge_voice(voice)
			if edge_voice:
				self._maybe_warn(f"Using Edge TTS voice={edge_voice}")
				try:
					return self._synthesize_edge(text, edge_voice, output_path)
				except Exception as exc:
					self._maybe_warn(f"Edge TTS failed: {exc}")
					if self._provider == "edge" or self._strict:
						raise
			else:
				self._maybe_warn("Edge TTS voice not resolved; using fallback.")
				if self._provider == "edge" or self._strict:
					raise RuntimeError("Edge TTS voice not configured.")
		elif self._provider == "edge" and self._strict:
			raise RuntimeError(self._edge_availability_reason() or "Edge TTS unavailable.")
		if self._can_use_piper():
			self._maybe_warn("Using Piper TTS")
			try:
				return self._synthesize_piper(text, voice, output_path)
			except Exception as exc:
				self._maybe_warn(f"Piper TTS failed: {exc}")
				if self._provider == "piper" or self._strict:
					raise
		elif self._provider == "piper" and self._strict:
			raise RuntimeError(self._piper_availability_reason() or "Piper TTS unavailable.")
		self._maybe_warn("Falling back to tone synthesis.")
		return self._synthesize_tone(text, voice, output_path)

	def _can_use_elevenlabs(self) -> bool:
		if self._provider not in {"auto", "elevenlabs"}:
			return False
		return bool(self._api_key and self._client)

	def _can_use_edge(self) -> bool:
		if self._provider not in {"auto", "edge"}:
			return False
		return self._edge_availability_reason() is None

	def _can_use_piper(self) -> bool:
		if self._provider not in {"auto", "piper"}:
			return False
		return self._piper_availability_reason() is None

	def _synthesize_elevenlabs(
		self, text: str, voice: VoiceProfile, output_path: Path, voice_id: str
	) -> int:
		if not self._output_format.startswith("wav_"):
			raise RuntimeError(
				"ELEVENLABS_OUTPUT_FORMAT must be a wav_* format for local WAV processing."
			)
		headers = {
			"xi-api-key": self._api_key,
			"accept": "audio/wav",
		}
		payload = {
			"text": text,
			"model_id": self._model_id,
			"voice_settings": voice.params or {},
		}
		response = self._client.post(
			f"/text-to-speech/{voice_id}",
			params={"output_format": self._output_format},
			headers=headers,
			json=payload,
		)
		if response.status_code >= 400:
			raise RuntimeError(self._format_error(response))
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_bytes(response.content)
		duration_ms = estimate_duration_ms(text)
		if output_path.suffix.lower() == ".wav":
			try:
				sample_rate, samples = read_wav(output_path)
				duration_ms = int(len(samples) / sample_rate * 1000)
			except ValueError:
				pass
		return duration_ms

	def _synthesize_edge(self, text: str, voice_id: str, output_path: Path) -> int:
		try:
			import edge_tts
		except ImportError as exc:
			raise RuntimeError("edge-tts is not installed. Add it to requirements.txt") from exc

		temp_path = output_path.with_name(f"{output_path.stem}_edge.wav")

		async def _run() -> None:
			communicate = edge_tts.Communicate(
				text,
				voice_id,
				rate=self._edge_rate,
				pitch=self._edge_pitch,
				volume=self._edge_volume,
				output_format=self._edge_output_format,
			)
			await communicate.save(str(temp_path))

		self._run_async(_run())
		self._resample_wav(temp_path, output_path, SAMPLE_RATE)
		temp_path.unlink(missing_ok=True)

		duration_ms = estimate_duration_ms(text)
		if output_path.suffix.lower() == ".wav":
			try:
				sample_rate, samples = read_wav(output_path)
				duration_ms = int(len(samples) / sample_rate * 1000)
			except ValueError:
				pass
		return duration_ms

	def _synthesize_piper(self, text: str, voice: VoiceProfile, output_path: Path) -> int:
		reason = self._piper_availability_reason()
		if reason is not None:
			raise RuntimeError(reason)

		model_path = Path(str(self._piper_model)).expanduser()
		config_path = self._resolve_piper_config_path()
		output_path.parent.mkdir(parents=True, exist_ok=True)

		cmd = [
			self._piper_bin,
			"--model",
			str(model_path),
			"--output_file",
			str(output_path),
		]
		if config_path is not None:
			cmd += ["--config", str(config_path)]
		speaker = self._resolve_piper_speaker(voice)
		if speaker is not None:
			cmd += ["--speaker", str(speaker)]

		if self._debug:
			print(f"[TTS] running: {' '.join(cmd)}", file=sys.stderr)

		try:
			result = subprocess.run(
				cmd,
				input=text.encode("utf-8"),
				capture_output=True,
				check=True,
			)
		except FileNotFoundError as exc:
			raise RuntimeError(f"Piper binary not found: {self._piper_bin}") from exc
		except subprocess.CalledProcessError as exc:
			stdout = exc.stdout.decode(errors="ignore") if exc.stdout else ""
			stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
			raise RuntimeError(
				"Piper TTS failed.\n"
				f"Command: {' '.join(cmd)}\n"
				f"Stdout: {stdout}\nStderr: {stderr}"
			) from exc

		if not output_path.exists() or output_path.stat().st_size == 0:
			stdout = result.stdout.decode(errors="ignore") if result.stdout else ""
			stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
			raise RuntimeError(
				"Piper produced no output.\n"
				f"Command: {' '.join(cmd)}\n"
				f"Stdout: {stdout}\nStderr: {stderr}"
			)

		duration_ms = estimate_duration_ms(text)
		if output_path.suffix.lower() == ".wav":
			try:
				sample_rate, samples = read_wav(output_path)
				duration_ms = int(len(samples) / sample_rate * 1000)
			except ValueError:
				pass
		return duration_ms

	def _synthesize_tone(self, text: str, voice: VoiceProfile, output_path: Path) -> int:
		duration_ms = estimate_duration_ms(text)
		frequency = self._voice_frequency(voice)
		samples = generate_tone_samples(duration_ms, frequency_hz=frequency, volume=0.2)
		write_wav(output_path, samples)
		return duration_ms

	def _edge_availability_reason(self) -> Optional[str]:
		try:
			import edge_tts  # noqa: F401
		except ImportError:
			return "edge-tts is not installed"
		return None

	def _resolve_edge_voice(self, voice: VoiceProfile) -> Optional[str]:
		params = voice.params or {}
		override = params.get("edge_voice") or params.get("edge_voice_id")
		if isinstance(override, str) and override.strip():
			return override.strip()
		gender = self._normalize_gender(voice.gender)
		if gender and self._edge_voice_by_gender.get(gender):
			value = self._edge_voice_by_gender.get(gender)
			if isinstance(value, str) and value.strip():
				return value.strip()
		if self._edge_voice_default and self._edge_voice_default.strip():
			return self._edge_voice_default.strip()
		return None

	def _resample_wav(self, src_path: Path, dest_path: Path, target_rate: int) -> None:
		with wave.open(str(src_path), "rb") as wav_file:
			channels = wav_file.getnchannels()
			sample_width = wav_file.getsampwidth()
			source_rate = wav_file.getframerate()
			frames = wav_file.readframes(wav_file.getnframes())

		if channels != 1 or sample_width != 2:
			raise RuntimeError("Edge TTS output must be 16-bit mono WAV.")
		if source_rate == target_rate:
			dest_path.write_bytes(src_path.read_bytes())
			return

		converted, _ = audioop.ratecv(
			frames,
			sample_width,
			channels,
			source_rate,
			target_rate,
			None,
		)
		samples = array("h")
		samples.frombytes(converted)
		write_wav(dest_path, samples)

	def _run_async(self, coro) -> None:
		try:
			loop = asyncio.get_running_loop()
		except RuntimeError:
			asyncio.run(coro)
			return
		if loop.is_running():
			new_loop = asyncio.new_event_loop()
			try:
				new_loop.run_until_complete(coro)
			finally:
				new_loop.close()
		else:
			loop.run_until_complete(coro)

	def _piper_availability_reason(self) -> Optional[str]:
		if not self._piper_model:
			return "PIPER_MODEL is not set"
		model_path = Path(str(self._piper_model)).expanduser()
		if not model_path.exists():
			return f"PIPER_MODEL not found: {model_path}"
		config_path = self._resolve_piper_config_path()
		if config_path is None:
			return "PIPER_CONFIG is not set and model JSON is missing"
		if not config_path.exists():
			return f"PIPER_CONFIG not found: {config_path}"
		if self._piper_bin and ("/" in self._piper_bin or "\\" in self._piper_bin):
			bin_path = Path(self._piper_bin).expanduser()
			if not bin_path.exists():
				return f"PIPER_BIN not found: {bin_path}"
		return None

	def _resolve_piper_config_path(self) -> Optional[Path]:
		if self._piper_config:
			return Path(str(self._piper_config)).expanduser()
		if not self._piper_model:
			return None
		model_path = Path(str(self._piper_model)).expanduser()
		candidate = model_path.with_suffix(model_path.suffix + ".json")
		if candidate.exists():
			return candidate
		return None

	def _resolve_piper_speaker(self, voice: VoiceProfile) -> Optional[str]:
		params = voice.params or {}
		for key in ("piper_speaker", "piper_speaker_id", "piper_speaker_idx"):
			value = params.get(key)
			if value is not None:
				return str(value)
		if self._piper_speaker:
			return str(self._piper_speaker)
		return None

	def _resolve_voice_id(
		self, voice: VoiceProfile, exclude_ids: Optional[set[str]] = None
	) -> Optional[str]:
		params = voice.params or {}
		override = params.get("voice_id") or params.get("voiceId")
		if isinstance(override, str) and override:
			return self._clean_voice_id(override, exclude_ids)

		name_override = params.get("voice_name") or params.get("voiceName")
		if isinstance(name_override, str) and name_override:
			by_name = self._pick_voice_id_by_name(name_override)
			if by_name:
				return self._clean_voice_id(by_name, exclude_ids)

		gender = self._normalize_gender(voice.gender)
		cache_key = self._cache_key(voice, gender)
		if cache_key in self._voice_cache:
			cached = self._clean_voice_id(self._voice_cache[cache_key], exclude_ids)
			if cached:
				return cached
		if gender and self._voice_id_by_gender.get(gender):
			voice_id = self._voice_id_by_gender[gender]
			if voice_id:
				cleaned = self._clean_voice_id(voice_id, exclude_ids)
				if cleaned:
					self._voice_cache[cache_key] = cleaned
					return cleaned

		voice_id = self._pick_voice_id_by_profile(voice)
		if voice_id:
			cleaned = self._clean_voice_id(voice_id, exclude_ids)
			if cleaned:
				self._voice_cache[cache_key] = cleaned
				return cleaned

		if self._voice_id_default:
			cleaned = self._clean_voice_id(self._voice_id_default, exclude_ids)
			if cleaned:
				return cleaned

		return self._clean_voice_id(self._pick_any_voice_id(), exclude_ids)

	def _pick_voice_id_by_name(self, name: str) -> Optional[str]:
		voices = self._fetch_voices()
		name_key = name.strip().lower()
		for voice in voices:
			voice_name = str(voice.get("name") or "").strip().lower()
			if voice_name == name_key:
				voice_id = self._extract_voice_id(voice)
				if voice_id:
					return voice_id
		return None

	def _pick_voice_id_by_profile(self, voice: VoiceProfile) -> Optional[str]:
		voices = self._fetch_voices()
		if not voices:
			return None

		target_gender = self._normalize_gender(voice.gender)
		target_style = self._normalize_text(voice.style)
		target_accent = self._normalize_text(voice.accent)
		target_age = self._normalize_text(voice.age)

		best_voice_id: Optional[str] = None
		best_score = -1
		for entry in voices:
			voice_id = self._extract_voice_id(entry)
			if not voice_id:
				continue
			score = self._score_voice(
				entry,
				target_gender,
				target_style,
				target_accent,
				target_age,
			)
			if score > best_score:
				best_score = score
				best_voice_id = voice_id
		return best_voice_id

	def _pick_any_voice_id(self) -> Optional[str]:
		voices = self._fetch_voices()
		if not voices:
			return None
		voice_id = self._extract_voice_id(voices[0])
		if voice_id:
			return voice_id
		return None

	def _fetch_voices(self) -> list[dict[str, object]]:
		if self._voice_catalog is not None:
			return self._voice_catalog
		if not self._client:
			self._voice_catalog = []
			return self._voice_catalog
		headers = {"xi-api-key": self._api_key}
		response = self._client.get("/voices", headers=headers)
		if response.status_code >= 400:
			raise RuntimeError(self._format_error(response))
		data = response.json()
		voices = data.get("voices", []) if isinstance(data, dict) else []
		self._voice_catalog = voices if isinstance(voices, list) else []
		return self._voice_catalog

	def _refresh_voice_catalog(self) -> None:
		self._voice_catalog = None
		self._voice_cache.clear()
		try:
			self._fetch_voices()
		except RuntimeError:
			pass

	@staticmethod
	def _extract_voice_id(voice: dict[str, object]) -> Optional[str]:
		voice_id = voice.get("voice_id") or voice.get("voiceId")
		if isinstance(voice_id, str) and voice_id:
			return voice_id.strip()
		return None

	def _score_voice(
		self,
		voice: dict[str, object],
		gender: Optional[str],
		style: Optional[str],
		accent: Optional[str],
		age: Optional[str],
	) -> int:
		score = 0
		labels = voice.get("labels") or {}
		label_gender = self._normalize_gender(labels.get("gender"))
		label_accent = self._normalize_text(labels.get("accent"))
		label_age = self._normalize_text(labels.get("age"))
		label_desc = self._normalize_text(labels.get("description"))
		voice_name = self._normalize_text(voice.get("name"))

		if gender and label_gender == gender:
			score += 5
		if accent and label_accent and accent in label_accent:
			score += 2
		if age and label_age and age in label_age:
			score += 1

		if style:
			keywords = STYLE_KEYWORDS.get(style, [style])
			if self._match_keywords(label_desc, keywords):
				score += 2
			if self._match_keywords(voice_name, keywords):
				score += 1

		return score

	@staticmethod
	def _match_keywords(text: Optional[str], keywords: list[str]) -> bool:
		if not text:
			return False
		return any(keyword in text for keyword in keywords)

	@staticmethod
	def _normalize_gender(value: object) -> Optional[str]:
		if not value:
			return None
		text = str(value).strip().lower()
		if text in {"female", "f", "woman", "girl"}:
			return "female"
		if text in {"male", "m", "man", "boy"}:
			return "male"
		if text in {"neutral", "nonbinary", "nb", "none"}:
			return "neutral"
		return None

	def _maybe_warn(self, message: str) -> None:
		if self._debug:
			print(f"[TTS] {message}", file=sys.stderr)

	@staticmethod
	def _format_error(response: httpx.Response) -> str:
		detail = response.text.strip()
		try:
			data = response.json()
		except ValueError:
			data = None

		if isinstance(data, dict):
			error = data.get("detail") or data.get("error")
			if isinstance(error, dict):
				message = error.get("message") or error.get("detail")
				if message:
					detail = message
			elif isinstance(error, str):
				detail = error

		return f"ElevenLabs API error {response.status_code}: {detail}"

	@staticmethod
	def _normalize_text(value: object) -> Optional[str]:
		if not value:
			return None
		return str(value).strip().lower()

	def _cache_key(self, voice: VoiceProfile, gender: Optional[str]) -> str:
		style = self._normalize_text(voice.style) or ""
		accent = self._normalize_text(voice.accent) or ""
		age = self._normalize_text(voice.age) or ""
		gender_key = gender or ""
		return f"{gender_key}|{style}|{accent}|{age}"

	@staticmethod
	def _clean_voice_id(value: Optional[str], exclude_ids: Optional[set[str]]) -> Optional[str]:
		if not value:
			return None
		cleaned = value.strip()
		if not cleaned:
			return None
		if exclude_ids and cleaned in exclude_ids:
			return None
		return cleaned

	@staticmethod
	def _voice_frequency(voice: VoiceProfile) -> float:
		style = (voice.style or "").lower()
		return STYLE_FREQUENCIES.get(style, 240.0)
