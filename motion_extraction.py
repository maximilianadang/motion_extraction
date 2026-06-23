"""Motion extraction via inverted, time-delayed frame overlay.

The technique (popularized by Posy's "Motion Extraction" video):

1. Take the video and a *duplicate* of it.
2. Invert the colors of the duplicate.
3. Set the duplicate's opacity to 50% and lay it over the original.
4. Offset the duplicate in time by a small number of frames.

Per pixel, the result is::

    out = (1 - alpha) * frame[t] + alpha * (255 - frame[t - delay])

With ``alpha = 0.5`` this simplifies to::

    out = 127.5 + 0.5 * (frame[t] - frame[t - delay])

So anything that did *not* move between ``t - delay`` and ``t`` cancels out to
neutral gray, while anything that moved leaves a visible, ghost-like edge. The
time ``delay`` is what makes motion appear: with ``delay = 0`` every pixel is
overlaid on its own inverse and the whole frame collapses to flat gray.

Frames are decoded with OpenCV and encoded as H.264 (yuv420p) by FFmpeg, so the
output plays anywhere Chromium does -- browsers, and VS Code video-preview
extensions. The FFmpeg binary is the static one bundled with imageio-ffmpeg, so
no system FFmpeg / sudo is required.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import imageio_ffmpeg


def extract_motion(
    input_path: Path,
    output_path: Path,
    delay: int = 1,
    alpha: float = 0.5,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Write a motion-extracted, H.264-encoded copy of ``input_path``."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"error: could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    proc: subprocess.Popen | None = None
    # Holds the previous `delay` frames; once full, history[0] is the frame
    # captured exactly `delay` frames ago.
    history: deque = deque(maxlen=delay)
    weight = 1.0 - alpha
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if proc is None:
                # Start FFmpeg once we know the decoded frame size (respects any
                # rotation the decoder applied, common for phone .MOV files).
                height, width = frame.shape[:2]
                proc = _start_ffmpeg(
                    ffmpeg_exe, output_path, width, height, fps, crf, preset
                )

            # During warm-up (before we have `delay` frames of history) overlay
            # the frame on itself, which produces neutral gray -> no false motion.
            delayed = history[0] if len(history) == delay else frame
            inverted = cv2.bitwise_not(delayed)
            blended = cv2.addWeighted(frame, weight, inverted, alpha, 0.0)

            assert proc.stdin is not None
            proc.stdin.write(blended.tobytes())

            history.append(frame)
            idx += 1
            if idx % 30 == 0:
                if total:
                    pct = 100.0 * idx / total
                    print(f"\r  {idx}/{total} frames ({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {idx} frames", end="", flush=True)
    finally:
        cap.release()
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise SystemExit(f"error: ffmpeg exited with status {ret}")

    print(f"\r  {idx} frames processed.{' ' * 20}")
    if idx == 0:
        raise SystemExit("error: no frames were read from the input.")
    print(f"Wrote {output_path}")


def _start_ffmpeg(
    ffmpeg_exe: str,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> subprocess.Popen:
    """Spawn an FFmpeg process that reads raw BGR frames and writes H.264."""
    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel", "error",
        # raw input: OpenCV frames are BGR uint8
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",  # no audio track
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",  # required for broad (Chromium/QuickTime) playback
        "-movflags", "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract motion by overlaying an inverted, time-delayed "
        "copy of each frame at reduced opacity. Output is H.264 mp4.",
    )
    parser.add_argument("input", type=Path, help="input video file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="output video file (default: <input>_motion.mp4)",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=int,
        default=1,
        help="frame offset between the original and the inverted copy "
        "(default: 1; larger = more pronounced trails, 0 = flat gray)",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help="opacity of the inverted overlay, 0..1 (default: 0.5)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality, 0=lossless..51=worst (default: 18, visually clean)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 speed/efficiency preset (default: medium)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input file does not exist: {args.input}")
    if args.delay < 1:
        parser.error("--delay must be >= 1 (0 produces a flat gray video)")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_motion.mp4")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        f"Extracting motion: {args.input} -> {args.output} "
        f"(delay={args.delay}, alpha={args.alpha}, crf={args.crf})"
    )
    extract_motion(
        args.input, args.output, args.delay, args.alpha, args.crf, args.preset
    )


if __name__ == "__main__":
    main(sys.argv[1:])
