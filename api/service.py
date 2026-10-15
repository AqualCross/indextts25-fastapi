"""TTS engine wrapper around the official IndexTTS-2.5 inference API."""

import logging
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".webm", ".opus")
PCM16_MAX = 32767.0
VOICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_VOICE_BYTES = 20 * 1024 * 1024


class VoiceNotFoundError(ValueError):
    """Raised when a voice id / audio path cannot be resolved."""


class TTSNotReadyError(RuntimeError):
    """Raised when the model is still loading."""


class TTSService:
    """Loads IndexTTS-2.5 and serializes GPU inference behind a lock."""

    def __init__(
        self,
        model_dir: str = "checkpoints",
        cfg_path: Optional[str] = None,
        device: Optional[str] = None,
        use_bf16: bool = True,
        use_qwen_emo: bool = False,
        voices_dir: str = "characters",
        use_deepspeed: bool = False,
        use_cuda_kernel: bool = False,
    ):
        self.model_dir = model_dir
        self.cfg_path = cfg_path or os.path.join(model_dir, "config.yaml")
        self.device = device
        self.use_bf16 = use_bf16
        self.use_qwen_emo = use_qwen_emo
        self.voices_dir = voices_dir
        self.use_deepspeed = use_deepspeed
        self.use_cuda_kernel = use_cuda_kernel

        os.makedirs(self.voices_dir, exist_ok=True)

        self.tts = None
        self.load_seconds: Optional[float] = None
        self.warmup_seconds: Optional[float] = None
        self.sample_rate = 22050  # native IndexTTS sample rate
        self._lock = threading.Lock()  # one inference at a time on the GPU

    # ------------------------------------------------------------------ loading

    @property
    def ready(self) -> bool:
        return self.tts is not None

    def load(self) -> None:
        """Construct the IndexTTS2 engine. Blocking; call once at startup."""
        if self.tts is not None:
            return

        # Imported lazily so that ``api.main`` can be imported (e.g. by tooling)
        # without pulling in torch.
        from indextts.infer_v2_5 import IndexTTS2

        start = time.perf_counter()
        logger.info(
            "Loading IndexTTS-2.5 (model_dir=%s, device=%s, bf16=%s, qwen_emo=%s)",
            self.model_dir, self.device, self.use_bf16, self.use_qwen_emo,
        )
        self.tts = IndexTTS2(
            cfg_path=self.cfg_path,
            model_dir=self.model_dir,
            use_bf16=self.use_bf16,
            device=self.device,
            use_cuda_kernel=self.use_cuda_kernel,
            use_deepspeed=self.use_deepspeed,
            use_qwen_emo=self.use_qwen_emo,
        )
        self.load_seconds = time.perf_counter() - start
        logger.info("Model loaded in %.1f s", self.load_seconds)

    def warmup(self, voice_path: Optional[str] = None, text: str = "你好，欢迎使用语音合成服务。") -> None:
        """Run one short inference to warm up CUDA kernels / caches."""
        if self.tts is None:
            raise TTSNotReadyError("Model is not loaded yet")
        if voice_path is None:
            voice_path = self._first_example_voice()
        if not voice_path or not os.path.isfile(voice_path):
            logger.warning("Warmup skipped: no reference audio available")
            return
        start = time.perf_counter()
        try:
            self.synthesize(
                text=text, voice_path=voice_path, lang="ZH",
                duration_factor=1.0, verbose=False,
            )
            self.warmup_seconds = time.perf_counter() - start
            logger.info("Warmup done in %.1f s", self.warmup_seconds)
        except Exception:  # noqa: BLE001 - warmup must never kill the server
            logger.exception("Warmup inference failed; continuing anyway")

    @staticmethod
    def _first_example_voice() -> Optional[str]:
        for name in ("voice_01.wav", "voice_02.wav", "voice_07.wav"):
            path = os.path.join("examples", name)
            if os.path.isfile(path):
                return path
        return None

    # ------------------------------------------------------------------ voices

    def list_voices(self) -> List[dict]:
        voices = []
        if os.path.isdir(self.voices_dir):
            for name in sorted(os.listdir(self.voices_dir)):
                base, ext = os.path.splitext(name)
                if ext.lower() in AUDIO_EXTS:
                    voices.append({
                        "id": base,
                        "path": os.path.join(self.voices_dir, name),
                    })
        return voices

    def resolve_voice(self, voice: str) -> str:
        """Map a voice id (or an explicit audio path) to an existing file path."""
        if not voice:
            raise VoiceNotFoundError("voice must not be empty")

        # Explicit path (absolute or relative) to an existing audio file.
        if os.path.isfile(voice) and os.path.splitext(voice)[1].lower() in AUDIO_EXTS:
            return os.path.abspath(voice)

        # Voice id inside the voices directory.
        for ext in AUDIO_EXTS:
            candidate = os.path.join(self.voices_dir, voice + ext)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        # Voice id that already carries its extension.
        candidate = os.path.join(self.voices_dir, voice)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

        available = ", ".join(v["id"] for v in self.list_voices()) or "<none>"
        raise VoiceNotFoundError(
            f"Voice '{voice}' not found. Available voices: {available}"
        )

    # ------------------------------------------------------------------ voice management

    def save_voice(self, name: str, data: bytes, ext: str) -> dict:
        """Store an uploaded reference audio as ``<voices_dir>/<name><ext>``.

        The file is validated with ffprobe first; an invalid upload is rejected
        and removed again.
        """
        if not VOICE_NAME_PATTERN.match(name or ""):
            raise ValueError(
                "voice name must be 1-64 chars of letters, digits, '_', '-' or '.' "
                "(it is used as a file name)"
            )
        ext = (ext or "").lower()
        if ext not in AUDIO_EXTS:
            raise ValueError(f"unsupported audio extension '{ext}', expected one of {', '.join(AUDIO_EXTS)}")
        if not data:
            raise ValueError("uploaded file is empty")
        if len(data) > MAX_VOICE_BYTES:
            raise ValueError(f"audio file too large ({len(data) / 1024 ** 2:.1f} MiB > 20 MiB)")

        # Drop other formats of the same voice id to keep one file per id.
        for other_ext in AUDIO_EXTS:
            if other_ext == ext:
                continue
            existing = os.path.join(self.voices_dir, name + other_ext)
            if os.path.isfile(existing):
                os.remove(existing)

        path = os.path.join(self.voices_dir, name + ext)
        tmp_path = path + ".part"
        with open(tmp_path, "wb") as fp:
            fp.write(data)
        try:
            duration = self.probe_duration(tmp_path)
        except Exception as exc:  # noqa: BLE001
            os.remove(tmp_path)
            raise ValueError(f"uploaded audio is not decodable ({exc})") from exc
        os.replace(tmp_path, path)

        logger.info("voice saved: %s (%.2f s)", path, duration)
        return {
            "id": name,
            "path": os.path.relpath(path),
            "duration_seconds": round(duration, 2),
        }

    def delete_voice(self, voice_id: str) -> List[str]:
        if not VOICE_NAME_PATTERN.match(voice_id or ""):
            raise ValueError("invalid voice id")
        removed = []
        for ext in AUDIO_EXTS:
            path = os.path.join(self.voices_dir, voice_id + ext)
            if os.path.isfile(path):
                os.remove(path)
                removed.append(os.path.relpath(path))
        if not removed:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")
        logger.info("voice deleted: %s", ", ".join(removed))
        return removed

    @staticmethod
    def probe_duration(path: str) -> float:
        """Return audio duration in seconds using ffprobe; raise on invalid input."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError("ffprobe not found in PATH")
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
        try:
            return float(proc.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("no audio stream found") from exc

    # ------------------------------------------------------------------ inference

    def synthesize(
        self,
        text: str,
        voice_path: str,
        lang: str = "ZH",
        emo_audio_path: Optional[str] = None,
        emo_alpha: float = 1.0,
        emo_vector: Optional[List[float]] = None,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,
        duration_factor: float = 1.0,
        text_normalization: bool = True,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        verbose: bool = False,
        **generation_kwargs,
    ) -> Tuple[int, np.ndarray]:
        """Run inference and return ``(sample_rate, int16 mono samples)``."""
        if self.tts is None:
            raise TTSNotReadyError("Model is not loaded yet")

        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with self._lock:
            result = self.tts.infer(
                spk_audio_prompt=voice_path,
                text=text,
                output_path=None,
                lang=lang,
                emo_audio_prompt=emo_audio_path,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                use_random=use_random,
                interval_silence=interval_silence,
                verbose=verbose,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                duration_factor=duration_factor,
                text_normalization=text_normalization,
                **generation_kwargs,
            )

        if result is None:
            raise RuntimeError("IndexTTS returned no audio")

        sample_rate, wav_data = result
        self.sample_rate = int(sample_rate)
        samples = np.asarray(wav_data)
        if samples.ndim == 2:  # (N, 1) -> (N,)
            samples = samples[:, 0]
        if samples.dtype != np.int16:
            samples = np.clip(samples, -PCM16_MAX, PCM16_MAX).astype(np.int16)
        return self.sample_rate, samples

    # ------------------------------------------------------------------ encoding

    @staticmethod
    def write_pcm_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
        with wave.open(path, "wb") as fp:
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp.setframerate(sample_rate)
            fp.writeframes(samples.astype(np.int16).tobytes())

    def encode(
        self,
        samples: np.ndarray,
        sample_rate: int,
        response_format: str = "wav",
        target_sample_rate: Optional[int] = None,
        gain_db: float = 0.0,
    ) -> bytes:
        """Apply gain / resampling and encode to the requested container."""
        target_sample_rate = target_sample_rate or sample_rate
        response_format = response_format.lower()

        # Raw little-endian PCM16: no container, no resampling.
        if response_format == "pcm":
            scale = 10 ** (gain_db / 20.0) if abs(gain_db) > 1e-6 else 1.0
            out = samples.astype(np.float32) * scale
            return np.clip(out, -PCM16_MAX, PCM16_MAX).astype(np.int16).tobytes()

        # Fast path: no post-processing needed.
        if (
            response_format == "wav"
            and target_sample_rate == sample_rate
            and abs(gain_db) < 1e-6
        ):
            import io

            buf = io.BytesIO()
            with wave.open(buf, "wb") as fp:
                fp.setnchannels(1)
                fp.setsampwidth(2)
                fp.setframerate(sample_rate)
                fp.writeframes(samples.astype(np.int16).tobytes())
            return buf.getvalue()

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for gain/resample/mp3/ogg conversion but was not found in PATH"
            )

        import tempfile

        with tempfile.TemporaryDirectory(prefix="indextts_api_") as tmp:
            in_path = os.path.join(tmp, "in.wav")
            out_path = os.path.join(tmp, f"out.{response_format}")
            self.write_pcm_wav(in_path, samples, sample_rate)

            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", in_path]
            if abs(gain_db) > 1e-6:
                cmd += ["-af", f"volume={gain_db}dB"]
            cmd += ["-ar", str(target_sample_rate)]
            if response_format == "mp3":
                cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
            elif response_format == "ogg":
                cmd += ["-c:a", "libvorbis", "-q:a", "5"]
            elif response_format == "wav":
                cmd += ["-c:a", "pcm_s16le"]
            else:
                raise ValueError(f"Unsupported response_format: {response_format}")
            cmd.append(out_path)

            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}"
                )
            with open(out_path, "rb") as fp:
                return fp.read()

    # ------------------------------------------------------------------ streaming synthesis

    @staticmethod
    def _to_int16(tensor) -> np.ndarray:
        """Convert a PCM-scale torch tensor (1, N) from the generator to int16 numpy."""
        import torch

        arr = tensor.detach().to(device="cpu", dtype=torch.float32).clamp_(-PCM16_MAX, PCM16_MAX)
        arr = arr.to(torch.int16).numpy()
        if arr.ndim == 2:  # (1, N) -> (N,)
            arr = arr[0]
        return arr

    def synthesize_stream(
        self,
        text: str,
        voice_path: str,
        lang: str = "ZH",
        emo_audio_path: Optional[str] = None,
        emo_alpha: float = 1.0,
        emo_vector: Optional[List[float]] = None,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,
        duration_factor: float = 1.0,
        text_normalization: bool = True,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        verbose: bool = False,
        **generation_kwargs,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Progressively synthesize speech, yielding ``(sample_rate, int16 mono)`` chunks.

        Audio is emitted as soon as each text segment has been generated, so for
        long inputs the client starts receiving audio well before the request
        finishes. The GPU lock is held for the whole iteration; closing the
        generator releases it.
        """
        if self.tts is None:
            raise TTSNotReadyError("Model is not loaded yet")

        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with self._lock:
            # Mirror the low-VRAM chunking of ``infer()`` so streaming never
            # generates a segment longer than the non-streaming path would.
            segments = [text]
            if getattr(self.tts, "low_vram", False) and len(text) > 40:
                segments = self.tts.split_text_by_punctuation(text, 40) or [text]

            for segment in segments:
                generator = self.tts.infer_generator(
                    spk_audio_prompt=voice_path,
                    text=segment,
                    output_path=None,
                    lang=lang,
                    emo_audio_prompt=emo_audio_path,
                    emo_alpha=emo_alpha,
                    emo_vector=emo_vector,
                    use_emo_text=use_emo_text,
                    emo_text=emo_text,
                    use_random=use_random,
                    interval_silence=interval_silence,
                    verbose=verbose,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    stream_return=True,
                    duration_factor=duration_factor,
                    text_normalization=text_normalization,
                    **generation_kwargs,
                )
                for chunk in generator:
                    # Upstream yields the segment waveform, then the inter-segment
                    # silence; both are PCM-scale tensors.
                    yield self.sample_rate, self._to_int16(chunk)

    def iter_encoded_stream(
        self,
        chunks: Iterable[Tuple[int, np.ndarray]],
        sample_rate: int,
        response_format: str = "wav",
        target_sample_rate: Optional[int] = None,
        gain_db: float = 0.0,
    ) -> Iterator[bytes]:
        """Encode streamed PCM chunks on the fly (ffmpeg pipe), yielding bytes.

        ``response_format='pcm'`` skips ffmpeg and emits raw 16-bit little-endian
        mono PCM at the model's native sample rate.
        """
        fmt = response_format.lower()

        if fmt == "pcm":
            for _sr, samples in chunks:
                yield samples.astype(np.int16, copy=False).tobytes()
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for streaming mp3/ogg/wav but was not found in PATH")

        target_sample_rate = target_sample_rate or sample_rate
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        ]
        if abs(gain_db) > 1e-6:
            cmd += ["-af", f"volume={gain_db}dB"]
        cmd += ["-ar", str(target_sample_rate)]
        if fmt == "mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"]
        elif fmt == "ogg":
            cmd += ["-c:a", "libvorbis", "-q:a", "5", "-f", "ogg"]
        elif fmt == "wav":
            cmd += ["-c:a", "pcm_s16le", "-f", "wav"]
        else:
            raise ValueError(f"Unsupported response_format: {response_format}")
        cmd.append("pipe:1")

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )

        stderr_lines: List[bytes] = []

        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, b""):
                    stderr_lines.append(line)
            except Exception:  # noqa: BLE001
                pass

        def _feed_stdin() -> None:
            try:
                for _sr, samples in chunks:
                    proc.stdin.write(samples.astype(np.int16, copy=False).tobytes())
                proc.stdin.close()
            except Exception:  # noqa: BLE001 - broken pipe on client disconnect etc.
                try:
                    chunks.close()  # type: ignore[attr-defined] - release the model lock
                except Exception:  # noqa: BLE001
                    pass
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=_drain_stderr, daemon=True).start()
        feeder = threading.Thread(target=_feed_stdin, daemon=True)
        feeder.start()

        try:
            while True:
                data = proc.stdout.read(64 * 1024)
                if not data:
                    break
                yield data
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            feeder.join(timeout=30)
            if proc.returncode not in (0, None) and stderr_lines:
                logger.warning(
                    "ffmpeg stream exited with code %s: %s",
                    proc.returncode,
                    b"".join(stderr_lines[-5:]).decode(errors="replace").strip(),
                )

    def vram_report(self) -> Optional[dict]:
        """Current + peak VRAM usage of this process, and device-level free memory."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            idx = torch.cuda.current_device()
            to_gib = lambda n: round(n / 1024 ** 3, 2)  # noqa: E731
            free, total = torch.cuda.mem_get_info(idx)
            return {
                "allocated_gib": to_gib(torch.cuda.memory_allocated(idx)),
                "reserved_gib": to_gib(torch.cuda.memory_reserved(idx)),
                "peak_allocated_gib": to_gib(torch.cuda.max_memory_allocated(idx)),
                "peak_reserved_gib": to_gib(torch.cuda.max_memory_reserved(idx)),
                "device_free_gib": to_gib(free),
                "device_total_gib": to_gib(total),
            }
        except Exception:  # noqa: BLE001
            return None
