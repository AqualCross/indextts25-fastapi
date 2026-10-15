"""Pydantic models for the IndexTTS-2.5 HTTP API.

The request body of ``POST /v1/audio/speech`` is a superset of the OpenAI
speech endpoint (and of csllpr/index-tts-fastapi), extended with the
IndexTTS-2.5 specific controls (language, emotion, duration factor, ...).
"""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class SpeechRequest(BaseModel):
    """Request model for speech synthesis (OpenAI-compatible)."""

    model: str = "IndexTTS-2.5"
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        ...,
        description=(
            "Voice identifier: a file name (without extension) inside the "
            "voices directory, e.g. 'alex' -> characters/alex.wav; an absolute "
            "path to an audio file is also accepted. OpenAI-style objects such "
            "as {\"id\": \"voice_123\"} are accepted as well."
        ),
    )
    response_format: Literal["mp3", "wav", "ogg", "pcm"] = Field(
        "mp3",
        description="pcm = raw 16-bit little-endian mono at the model's native sample rate (streaming friendly)",
    )
    sample_rate: int = Field(24000, ge=8000, le=48000, description="Ignored for response_format=pcm")
    stream: bool = Field(
        False,
        description=(
            "Progressive streaming: audio bytes are returned while the remaining "
            "segments are still being generated (mp3/ogg/wav are piped through ffmpeg, "
            "pcm is passed through untouched)."
        ),
    )
    speed: float = Field(
        1.0, gt=0.0, le=3.0,
        description="1.0 = normal. Mapped to duration_factor = 1/speed; "
                    "ignored when duration_factor is explicitly provided.",
    )
    gain: float = Field(0.0, ge=-30.0, le=30.0, description="Output gain in dB")

    # ---- IndexTTS-2.5 extensions -------------------------------------------------
    lang: str = Field("ZH", description="Language code: ZH / EN / JA / ES / AR")
    duration_factor: Optional[float] = Field(
        None, ge=0.5, le=2.0,
        description="Speaking speed: >1.0 slower, <1.0 faster (2.5 native control).",
    )
    emo_audio: Optional[str] = Field(
        None,
        description="Emotion reference: voice id or audio path (prompt voice is used when omitted).",
    )
    emo_alpha: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Strength of the emotion reference (only used with emo_audio).",
    )
    emo_vector: Optional[List[float]] = Field(
        None, min_length=8, max_length=8,
        description="Emotion intensities in order "
                    "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm], each 0..1.",
    )
    use_emo_text: bool = Field(
        False,
        description="Infer the emotion from the text (requires the QwenEmotion model, see --qwen-emo).",
    )
    emo_text: Optional[str] = Field(
        None,
        description="Explicit emotion description; only used when use_emo_text=true.",
    )
    use_random: bool = Field(False, description="Random emotion sampling (only with emo_vector).")
    text_normalization: bool = Field(True, description="Normalize numbers/units/... in the input text.")
    interval_silence: int = Field(200, ge=0, le=2000, description="Silence between text segments, in ms.")
    max_text_tokens_per_segment: int = Field(
        120, ge=10, le=1000,
        description="Max text tokens per generated segment (longer text is split automatically).",
    )

    # ---- generation knobs (passed to the GPT sampler) ----------------------------
    do_sample: Optional[bool] = Field(None, description="Sample instead of greedy decoding (official default: true)")
    num_beams: Optional[int] = Field(
        None, ge=1, le=8,
        description="Beam width of the GPT stage (official default: 3). 1 is the fastest, with a small quality trade-off.",
    )

    # ---- compatibility layer ------------------------------------------------------
    extra_params: Optional[dict] = Field(
        None,
        description=(
            "Compatibility layer for iamyb/indextts-api & vLLM-Omni clients: model-specific "
            "parameters (lang, emo_vector, emo_alpha, emo_text, use_emo_text, emo_audio, "
            "use_random, text_normalization, duration_factor, ...) which take precedence "
            "over the flat fields."
        ),
    )

    @field_validator("voice", mode="before")
    @classmethod
    def _unpack_openai_voice_object(cls, value):
        """Accept OpenAI's object form, e.g. {"id": "voice_123"}."""
        if isinstance(value, dict):
            return value.get("id") or value.get("voice") or value.get("name") or ""
        return value
    temperature: Optional[float] = Field(None, gt=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=200)
    repetition_penalty: Optional[float] = Field(None, gt=0.0, le=20.0)
    max_mel_tokens: Optional[int] = Field(None, ge=64, le=4000)
