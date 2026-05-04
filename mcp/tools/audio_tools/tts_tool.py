from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import httpx

from shared.schemas.state import VoiceProfile

from .audio_utils import estimate_duration_ms, generate_tone_samples, read_wav, write_wav


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
		self._provider = provider or os.getenv("TTS_PROVIDER", "auto")
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
		self._maybe_warn("Falling back to tone synthesis.")
		return self._synthesize_tone(text, voice, output_path)

	def _can_use_elevenlabs(self) -> bool:
		if self._provider not in {"auto", "elevenlabs"}:
			return False
		return bool(self._api_key and self._client)

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

	def _synthesize_tone(self, text: str, voice: VoiceProfile, output_path: Path) -> int:
		duration_ms = estimate_duration_ms(text)
		frequency = self._voice_frequency(voice)
		samples = generate_tone_samples(duration_ms, frequency_hz=frequency, volume=0.2)
		write_wav(output_path, samples)
		return duration_ms

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
