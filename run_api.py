"""Launcher for the IndexTTS-2.5 FastAPI server.

Examples:
    uv run run_api.py --gpu 1 --port 8000
    uv run run_api.py --device cuda:0 --no-bf16 --qwen-emo
    uv run run_api.py --api-token secret --voice-dir characters
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IndexTTS-2.5 API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="Physical GPU index to use (sets CUDA_VISIBLE_DEVICES before torch starts)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device string, e.g. cuda:0 / cpu (overrides --gpu semantics; default: cuda:0 if available)",
    )
    parser.add_argument("--model-dir", type=str, default="checkpoints", help="Model directory (default: checkpoints)")
    parser.add_argument("--cfg-path", type=str, default=None, help="config.yaml path (default: <model-dir>/config.yaml)")
    parser.add_argument("--voices-dir", type=str, default="characters", help="Directory with reference voices (default: characters)")
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=True, help="Use BF16 inference (default: on)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable BF16 (use FP32)")
    parser.add_argument(
        "--qwen-emo", dest="qwen_emo", action="store_true", default=False,
        help="Load the QwenEmotion model (needed for use_emo_text; ~1.2 GiB extra VRAM)",
    )
    parser.add_argument("--no-qwen-emo", dest="qwen_emo", action="store_false", help="Do not load QwenEmotion (default)")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True, help="Run one warmup inference after loading (default: on)")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Skip the warmup inference")
    parser.add_argument("--api-token", type=str, default=None, help="Require this Bearer token on /v1/* (default: no auth)")
    parser.add_argument("--log-level", type=str, default="info", help="uvicorn log level (default: info)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.gpu is not None and args.device is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.device = "cuda:0"

    os.environ["INDEXTTS_MODEL_DIR"] = args.model_dir
    if args.cfg_path:
        os.environ["INDEXTTS_CFG_PATH"] = args.cfg_path
    if args.device:
        os.environ["INDEXTTS_DEVICE"] = args.device
    os.environ["INDEXTTS_BF16"] = "1" if args.bf16 else "0"
    os.environ["INDEXTTS_QWEN_EMO"] = "1" if args.qwen_emo else "0"
    os.environ["INDEXTTS_VOICES_DIR"] = args.voices_dir
    os.environ["INDEXTTS_WARMUP"] = "1" if args.warmup else "0"
    if args.api_token:
        os.environ["INDEXTTS_API_TOKEN"] = args.api_token

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: uv sync --extra api", file=sys.stderr)
        return 1

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
