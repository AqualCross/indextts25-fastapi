"""FastAPI application exposing IndexTTS-2.5 over HTTP.

API surface (OpenAI speech compatible; mirrors the endpoint layout of
csllpr/index-tts-fastapi, iamyb/indextts-api and the vLLM-Omni recipe):

    GET    /health                  - liveness + model status (incl. VRAM)
    GET    /v1/models               - OpenAI-compatible model list
    GET    /v1/voices               - list voices (alias: /v1/audio/voices)
    POST   /v1/voices               - upload a reference audio as a voice
    DELETE /v1/voices/{voice_id}    - delete a voice
    POST   /v1/audio/speech         - synthesize speech (stream=true => progressive audio)
    POST   /v1/audio/speech/stream  - synthesize speech as base64 PCM16 SSE events

The model is loaded in a background thread at startup, so the server answers
``/health`` immediately while the (possibly multi-minute) load is running.
"""

import asyncio
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Iterator, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from .models import SpeechRequest
from .service import MAX_VOICE_BYTES, TTSNotReadyError, TTSService, VoiceNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("indextts.api")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- configuration

MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH = os.environ.get("INDEXTTS_CFG_PATH") or os.path.join(MODEL_DIR, "config.yaml")
DEVICE = os.environ.get("INDEXTTS_DEVICE") or None
USE_BF16 = _env_bool("INDEXTTS_BF16", True)
USE_QWEN_EMO = _env_bool("INDEXTTS_QWEN_EMO", False)
VOICES_DIR = os.environ.get("INDEXTTS_VOICES_DIR", "characters")
API_TOKEN = os.environ.get("INDEXTTS_API_TOKEN", "").strip()
RUN_WARMUP = _env_bool("INDEXTTS_WARMUP", True)

CONTENT_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "pcm": "audio/pcm"}

service = TTSService(
    model_dir=MODEL_DIR,
    cfg_path=CFG_PATH,
    device=DEVICE,
    use_bf16=USE_BF16,
    use_qwen_emo=USE_QWEN_EMO,
    voices_dir=VOICES_DIR,
)


def _load_model_background() -> None:
    try:
        service.load()
        if RUN_WARMUP:
            service.warmup()
    except Exception:  # noqa: BLE001 - keep the server alive, report via /health
        logger.exception("Failed to load IndexTTS model")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Starting IndexTTS API server (model_dir=%s, device=%s, bf16=%s, qwen_emo=%s)",
        MODEL_DIR, DEVICE or "auto", USE_BF16, USE_QWEN_EMO,
    )
    loader = asyncio.get_running_loop().run_in_executor(None, _load_model_background)
    yield
    try:
        await asyncio.wait_for(loader, timeout=5)
    except Exception:  # noqa: BLE001
        pass
    logger.info("IndexTTS API server stopped")


app = FastAPI(
    title="IndexTTS-2.5 API",
    description="OpenAI-compatible speech synthesis API for IndexTTS-2.5 (BF16)",
    version="2.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Static bearer-token check. Disabled when no token is configured."""
    if not API_TOKEN:
        return
    token = credentials.credentials if credentials else ""
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ------------------------------------------------------------------- helpers


def _require_ready() -> None:
    if not service.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading, retry later (see /health)",
        )


def _effective_request(request: SpeechRequest) -> SpeechRequest:
    """Compatibility layer: ``extra_params`` (iamyb/vLLM style) overrides flat fields."""
    if not request.extra_params:
        return request
    allowed = {
        key: value
        for key, value in request.extra_params.items()
        if key in SpeechRequest.model_fields and key != "extra_params"
    }
    if allowed:
        logger.debug("extra_params overrides: %s", ", ".join(allowed))
    return request.model_copy(update=allowed)


def _prepare_synthesis(request: SpeechRequest) -> Tuple[SpeechRequest, dict]:
    """Apply extra_params, resolve voices and build the service kwargs."""
    params = _effective_request(request)

    # ``voice`` may be "IndexTTS:alex" like in index-tts-fastapi.
    _, voice = params.voice.split(":", 1) if ":" in params.voice else ("", params.voice)

    try:
        voice_path = service.resolve_voice(voice)
        emo_audio_path = service.resolve_voice(params.emo_audio) if params.emo_audio else None
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    duration_factor = params.duration_factor
    if duration_factor is None:
        duration_factor = 1.0 / params.speed

    synth_kwargs = dict(
        voice_path=voice_path,
        lang=params.lang,
        emo_audio_path=emo_audio_path,
        emo_alpha=params.emo_alpha,
        emo_vector=params.emo_vector,
        use_emo_text=params.use_emo_text,
        emo_text=params.emo_text,
        use_random=params.use_random,
        duration_factor=duration_factor,
        text_normalization=params.text_normalization,
        interval_silence=params.interval_silence,
        max_text_tokens_per_segment=params.max_text_tokens_per_segment,
        do_sample=params.do_sample,
        num_beams=params.num_beams,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        repetition_penalty=params.repetition_penalty,
        max_mel_tokens=params.max_mel_tokens,
    )
    return params, synth_kwargs


def _audio_response_meta(fmt: str, content_length: Optional[int] = None) -> Tuple[str, dict]:
    fmt = fmt.lower()
    media_type = CONTENT_TYPES.get(fmt, "application/octet-stream")
    headers = {
        "Content-Disposition": f"attachment; filename=speech.{fmt}",
        "Cache-Control": "no-store",
    }
    if fmt == "pcm":
        headers.update({
            "X-Sample-Rate": str(service.sample_rate),
            "X-Channels": "1",
            "X-Format": "s16le",
        })
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return media_type, headers


# --------------------------------------------------------------------- routes


@app.get("/health")
async def health_check():
    vram = service.vram_report()
    payload = {
        "status": "ok" if service.ready else "loading",
        "model_loaded": service.ready,
        "model_dir": MODEL_DIR,
        "device": DEVICE or "auto",
        "bf16": USE_BF16,
        "qwen_emo": USE_QWEN_EMO,
        "voices": len(service.list_voices()),
        "load_seconds": service.load_seconds,
        "warmup_seconds": service.warmup_seconds,
    }
    if vram:
        payload["vram"] = vram
    return JSONResponse(payload, status_code=200 if service.ready else 503)


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "IndexTTS-2.5",
                "object": "model",
                "created": 0,
                "owned_by": "index-tts",
            }
        ],
    }


# --------------------------------------------------------------- voice management


@app.get("/v1/voices", dependencies=[Depends(verify_token)])
@app.get("/v1/audio/voices", dependencies=[Depends(verify_token)])
async def list_voices():
    return {"object": "list", "data": service.list_voices()}


@app.post("/v1/voices", dependencies=[Depends(verify_token)], status_code=status.HTTP_201_CREATED)
async def upload_voice(
    file: UploadFile = File(..., description="Reference audio (wav/mp3/flac/ogg/..., <= 20 MiB)"),
    name: str = Form(..., description="Voice id used later in the 'voice' field (letters/digits/_/-)"),
    consent: Optional[str] = Form(None, description="Optional consent declaration, logged server-side"),
):
    data = await file.read(MAX_VOICE_BYTES + 1)
    ext = os.path.splitext(file.filename or "")[1].lower()
    try:
        info = await run_in_threadpool(service.save_voice, name, data, ext)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if consent:
        logger.info("voice upload '%s' consent: %s", name, consent[:120])
    return info


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(verify_token)])
async def delete_voice(voice_id: str):
    try:
        removed = await run_in_threadpool(service.delete_voice, voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"object": "voice.deleted", "id": voice_id, "removed": removed}


# --------------------------------------------------------------------- speech


@app.post("/v1/audio/speech", dependencies=[Depends(verify_token)])
async def generate_speech(request: SpeechRequest):
    _require_ready()
    params, synth_kwargs = _prepare_synthesis(request)

    logger.info(
        "speech request: voice=%s lang=%s format=%s stream=%s text=%r",
        os.path.basename(synth_kwargs["voice_path"]), params.lang, params.response_format,
        params.stream, params.input[:60],
    )

    if params.stream:
        chunks = service.synthesize_stream(text=params.input, **synth_kwargs)
        body = service.iter_encoded_stream(
            chunks,
            service.sample_rate,
            params.response_format,
            params.sample_rate,
            params.gain,
        )
        media_type, headers = _audio_response_meta(params.response_format)
        headers["X-Accel-Buffering"] = "no"
        return StreamingResponse(body, media_type=media_type, headers=headers)

    try:
        sample_rate, samples = await run_in_threadpool(
            service.synthesize, text=params.input, **synth_kwargs
        )
        audio = await run_in_threadpool(
            service.encode,
            samples,
            sample_rate,
            params.response_format,
            params.sample_rate,
            params.gain,
        )
    except (TTSNotReadyError, VoiceNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech synthesis failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    media_type, headers = _audio_response_meta(params.response_format, content_length=len(audio))
    return Response(content=audio, media_type=media_type, headers=headers)


@app.post("/v1/audio/speech/stream", dependencies=[Depends(verify_token)])
async def generate_speech_sse(request: SpeechRequest):
    """Server-Sent Events variant: metadata + base64 PCM16 chunks.

    Event payloads (JSON in ``data:`` lines):
      ``{"type": "start", "sample_rate": 22050, "format": "pcm16le", "channels": 1}``
      ``{"type": "audio", "audio": "<base64>", "sample_rate": 22050}``
      ``{"type": "done"}`` / ``{"type": "error", "message": "..."}``
    """
    _require_ready()
    params, synth_kwargs = _prepare_synthesis(request)

    logger.info(
        "speech SSE request: voice=%s lang=%s text=%r",
        os.path.basename(synth_kwargs["voice_path"]), params.lang, params.input[:60],
    )

    def event_stream() -> Iterator[str]:
        yield "retry: 1000\n\n"
        yield "data: " + json.dumps({
            "type": "start",
            "sample_rate": service.sample_rate,
            "format": "pcm16le",
            "channels": 1,
        }) + "\n\n"
        try:
            for sample_rate, samples in service.synthesize_stream(text=params.input, **synth_kwargs):
                chunk = samples.astype("<i2", copy=False).tobytes()
                yield "data: " + json.dumps({
                    "type": "audio",
                    "sample_rate": sample_rate,
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }) + "\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as exc:  # noqa: BLE001 - the stream already started; report inline
            logger.exception("SSE synthesis failed")
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
