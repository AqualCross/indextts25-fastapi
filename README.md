# indextts25-fastapi

OpenAI-compatible **HTTP API server** for [IndexTTS‑2.5](https://github.com/index-tts/index-tts) —
the multilingual zero-shot voice-cloning TTS model by Bilibili's Index Team.

This is a **thin wrapper**: it contains no upstream code and no model weights.
You deploy `index-tts` first (per its official instructions), drop this folder in,
and get a production-friendly HTTP service.

```
✅ OpenAI-compatible POST /v1/audio/speech (works with the OpenAI SDK)
✅ True streaming: bytes flow while later text segments are still generating
   (`stream: true`) + a base64-PCM16 SSE endpoint
✅ Voice management over HTTP: upload / list / delete reference voices
✅ Full IndexTTS-2.5 controls: lang, emotion vector / emotion text / emotion
   audio, duration_factor, pronunciation annotations, sampling knobs
✅ mp3 / wav / ogg (ffmpeg) and raw PCM output, speed / gain / sample_rate
✅ BF16 inference, background model load (health endpoint up immediately),
   GPU serialization, live VRAM stats (current + peak)
✅ Compatible with request bodies written for iamyb/indextts-api and the
   vLLM-Omni recipe (via `extra_params`)
✅ Tested on Windows + RTX 3070 8 GB (low-VRAM auto-chunking); Linux works too
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (the upstream project uses it) + Python 3.10+
- NVIDIA GPU with BF16 (8 GB VRAM is enough)
- `ffmpeg` / `ffprobe` available in `PATH` (for mp3/ogg, resampling, gain)

## Recommended layout

```
index-tts/                    ← upstream clone (run commands from here)
├── checkpoints/              ← model weights (per upstream README)
├── characters/               ← reference voices (<voice id>.wav), optional
└── indextts25-fastapi/       ← this repository, cloned inside
```

## Quick start

```bash
# 1) deploy upstream index-tts (deps + ~5.7 GB of weights)
git clone https://github.com/index-tts/index-tts.git && cd index-tts
uv sync
#    download weights into ./checkpoints — see upstream README (hf / modelscope):
#    uv run python -c "from indextts.utils.model_download import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', 'checkpoints')"

# 2) clone this wrapper inside and add its dependencies to the upstream env
git clone https://github.com/AqualCross/indextts25-fastapi.git
uv add fastapi "uvicorn[standard]" python-multipart

# 3) start the server (always run with cwd = index-tts root)
uv run indextts25-fastapi/run_api.py --gpu 0 --bf16 --port 8000
# Windows one-click:  indextts25-fastapi\start_api.bat
```

The model loads in a background thread — `GET /health` answers `503 "loading"`
immediately and turns into `200 "ok"` once ready (first start also downloads
the auxiliary models into `checkpoints/hf_cache/`).

> No reference voice yet? Fetch the upstream example clips and use one as a voice:
> ```bash
> uv run python -c "from indextts.utils.examples_downloader import ensure_examples_available; ensure_examples_available()"
> mkdir -p characters && cp examples/voice_01.wav characters/voice_01.wav
> ```
> then call the API with `"voice": "voice_01"`.

> Keep the working directory at the **index-tts root**: the upstream code
> resolves `checkpoints/`, `characters/`, `examples/` relative to the CWD.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | model status, device, VRAM (current + peak) |
| GET | `/v1/models` | OpenAI-compatible model list |
| GET | `/v1/voices` (alias `/v1/audio/voices`) | list voices |
| POST | `/v1/voices` | upload a voice (`file` + `name` [+ `consent`]), ffprobe-validated, ≤ 20 MiB |
| DELETE | `/v1/voices/{voice_id}` | delete a voice |
| POST | `/v1/audio/speech` | synthesize (`stream: true` = progressive audio) |
| POST | `/v1/audio/speech/stream` | SSE: `start` / `audio` (base64 PCM16) / `done` / `error` |

### `POST /v1/audio/speech` — main request fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `input` | string | required | text (ZH / EN / JA / ES / AR) |
| `voice` | string | required | voice id in the voices dir, a file path, or `IndexTTS:voice_id` |
| `response_format` | string | `mp3` | `mp3` / `wav` / `ogg` / `pcm` (raw 16-bit LE mono @ model rate) |
| `sample_rate` | int | `24000` | output rate (native 22050; ignored for `pcm`) |
| `stream` | bool | `false` | progressive streaming |
| `speed` | float | `1.0` | mapped to `duration_factor = 1/speed` |
| `gain` | float | `0.0` | dB |
| `lang` | string | `ZH` | `ZH` / `EN` / `JA` / `ES` / `AR` |
| `duration_factor` | float? | `null` | 0.5–2.0 (>1 slower), overrides `speed` |
| `emo_audio` / `emo_alpha` | string? / float | `null` / `1.0` | emotion reference voice + strength |
| `emo_vector` | float[8]? | `null` | `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` |
| `use_emo_text` / `emo_text` | bool / string? | `false` / `null` | text-driven emotion (needs server `--qwen-emo`) |
| `use_random`, `text_normalization`, `interval_silence`, `max_text_tokens_per_segment` | - | `false, true, 200, 120` | |
| `do_sample`, `num_beams`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_mel_tokens` | - | `null` (upstream defaults) | `num_beams=1` ≈ +15% speed |
| `extra_params` | object? | `null` | iamyb/vLLM-Omni compatibility layer, overrides flat fields |

Pronunciation control works inline in `input`, e.g.
`他在银<行|XING2>里<行|HANG2>走了半天`, `a <minute|M IH1 . N AH0 T> to remember`,
`<上手|じょうず>`.

## Examples

```bash
# synthesize (server-side request file avoids quoting pain on Windows)
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  --data-binary @examples/request.json -o out.mp3

# manage voices
curl -X POST http://127.0.0.1:8000/v1/voices \
  -F "file=@my_reference.wav" -F "name=my_voice"
curl  http://127.0.0.1:8000/v1/voices
curl -X DELETE http://127.0.0.1:8000/v1/voices/my_voice
```

```python
# OpenAI SDK — drop-in
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
client.audio.speech.create(model="IndexTTS-2.5", voice="voice_01",
                           input="Hello from IndexTTS 2.5!").stream_to_file("out.mp3")

# streaming: consume bytes while the rest is still generating
import requests
with requests.post("http://127.0.0.1:8000/v1/audio/speech", json={
    "input": "长文本……", "voice": "voice_01",
    "response_format": "mp3", "stream": True,
}, stream=True, timeout=300) as resp:
    with open("out.mp3", "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)

# bundled helpers
#   python client_example.py --text "你好" --voice voice_01 --output out.mp3
#   python client_example.py --text "长文本" --voice voice_01 --stream
```

SSE variant (`/v1/audio/speech/stream`): events are `{"type":"start",...}` →
`{"type":"audio","audio":"<base64 pcm16le>","sample_rate":22050}` → `{"type":"done"}`,
handy for browsers (`fetch` + `ReadableStream`).

## Startup options (`run_api.py` / `start_api.bat`)

```
--gpu N              physical GPU index (sets CUDA_VISIBLE_DEVICES)
--device cuda:0      torch device string
--bf16 / --no-bf16   BF16 inference (default: on)
--qwen-emo           load QwenEmotion (~1.2 GB extra VRAM; enables use_emo_text)
--api-token TOKEN    require Authorization: Bearer TOKEN on /v1/*
--model-dir DIR      weights dir (default checkpoints)
--voices-dir DIR     voice directory (default characters)
--no-warmup          skip the warm-up inference after loading
```

## Measured performance (RTX 3070 8 GB, BF16, Windows)

| Scenario | Audio | Wall time | RTF |
| --- | --- | --- | --- |
| 26-char sentence (`num_beams=1`) | 6.06 s | 7.6 s | 1.26 |
| 26-char sentence (default `num_beams=3`) | 6.30 s | 9.2 s | 1.45 |
| 86-char text (auto-split into 2 segments) | 16.0 s | 22.0 s | 1.37 |
| **streaming**: time to first playable audio, same 86-char text | — | **7.2 s** (vs 14.6 s non-stream) | — |

VRAM: idle ≈ 5.1 GiB allocated / 5.5 GiB reserved (model weights dominate);
a request adds only ~0.2–0.3 GiB peak. The reserved pool is returned to the
cache after each request (not a leak) — watch `GET /health` → `vram`.

## Notes

- One inference at a time per process (GPU lock); concurrent HTTP requests queue.
- First request per voice re-encodes the reference audio (~3 s extra); repeating
  requests with the same voice reuse the cache.
- The generated audio never touches the GPU after a request finishes; only the
  weights stay resident.
- `--qwen-emo` on an 8 GB card next to other GPU workloads can be tight.

## License & acknowledgements

The wrapper code in this repository is released under the [MIT License](LICENSE).

IndexTTS‑2.5 itself (code and model weights) is licensed separately by the
Bilibili Index Team — see the upstream
[LICENSE](https://github.com/index-tts/index-tts/blob/main/LICENSE) and
[DISCLAIMER](https://github.com/index-tts/index-tts/blob/main/DISCLAIMER).
This repository does not redistribute upstream code or weights.

---

## 中文说明（简）

这是 IndexTTS‑2.5 的 HTTP API 薄封装：**不含官方代码和模型权重**，需要先按官方文档部署
`index-tts`（`uv sync` + 下载权重），再把本仓库克隆进 `index-tts` 目录使用。

```bash
cd index-tts
uv add fastapi "uvicorn[standard]" python-multipart
git clone https://github.com/AqualCross/indextts25-fastapi.git
uv run indextts25-fastapi/run_api.py --gpu 0 --bf16 --port 8000   # 或 Windows 双击 start_api.bat
```

- 接口：OpenAI 兼容 `/v1/audio/speech`（支持 `stream: true` 真流式）+ `/v1/voices` 音色管理 + SSE
  `/v1/audio/speech/stream`
- 扩展参数：`lang`、`emo_vector`、`emo_text`、`emo_audio`、`duration_factor`、`num_beams` 等；
  另支持 `extra_params` 兼容层（iamyb / vLLM-Omni 风格请求体可直接用）
- 音色：把参考音频放到 `characters/<名字>.wav`，或用 `POST /v1/voices` 上传
- 8G 显存实测可用（BF16，low-VRAM 自动分段），RTX 3070 上 RTF ≈ 1.3–1.5，流式首段音频约为
  非流式的一半延迟
