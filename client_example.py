"""Minimal client for the IndexTTS-2.5 HTTP API (stdlib only).

Usage:
    python client_example.py --text "你好，世界" --voice voice_01 --output out.mp3
    python client_example.py --text "Hello" --voice voice_01 --lang EN --format wav \
        --emo-vector 0,0,0.8,0,0,0,0,0 --url http://127.0.0.1:8000
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the IndexTTS-2.5 /v1/audio/speech endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--token", default=None, help="Bearer token (only if the server requires one)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", required=True, help="Voice id (file name in the voices directory)")
    parser.add_argument("--output", default="output.mp3", help="Output file path")
    parser.add_argument("--format", default="mp3", choices=["mp3", "wav", "ogg"])
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--lang", default="ZH", help="ZH / EN / JA / ES / AR")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = normal")
    parser.add_argument("--gain", type=float, default=0.0, help="dB")
    parser.add_argument("--emo-audio", default=None, help="Emotion reference voice id")
    parser.add_argument("--emo-alpha", type=float, default=1.0)
    parser.add_argument(
        "--emo-vector", default=None,
        help="8 comma-separated values: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm",
    )
    parser.add_argument("--duration-factor", type=float, default=None, help=">1 slower, <1 faster")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    payload = {
        "model": "IndexTTS-2.5",
        "input": args.text,
        "voice": args.voice,
        "response_format": args.format,
        "sample_rate": args.sample_rate,
        "speed": args.speed,
        "gain": args.gain,
        "lang": args.lang,
        "stream": args.stream,
    }
    if args.emo_audio:
        payload["emo_audio"] = args.emo_audio
        payload["emo_alpha"] = args.emo_alpha
    if args.emo_vector:
        values = [float(v) for v in args.emo_vector.split(",")]
        if len(values) != 8:
            print("--emo-vector needs exactly 8 values", file=sys.stderr)
            return 2
        payload["emo_vector"] = values
    if args.duration_factor is not None:
        payload["duration_factor"] = args.duration_factor

    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if args.token:
        request.add_header("Authorization", f"Bearer {args.token}")

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            if args.stream:
                # Write progressively while the server keeps generating.
                total = 0
                with open(args.output, "wb") as fp:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        fp.write(chunk)
                        total += len(chunk)
                        fp.flush()
                audio = b""  # already written
                print(f"Saved {total} bytes (streamed) to {args.output}")
                return 0
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection failed: {exc.reason}", file=sys.stderr)
        print("Is the server running? Check GET /health.", file=sys.stderr)
        return 1

    with open(args.output, "wb") as fp:
        fp.write(audio)
    print(f"Saved {len(audio)} bytes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
