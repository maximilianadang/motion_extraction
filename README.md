# motion-extraction

Extract motion from a video using one simple technique: overlay each frame with
an **inverted, time-delayed, 50%-opacity copy of itself**.

Per pixel the output is:

```
out = (1 - alpha) * frame[t] + alpha * (255 - frame[t - delay])
```

With `alpha = 0.5` that's `127.5 + 0.5 * (frame[t] - frame[t - delay])`: parts of
the scene that didn't move cancel to neutral gray, while anything that moved
leaves a visible edge. The **time `delay`** is what reveals motion — with
`delay = 0` every pixel cancels against its own inverse and the whole frame goes
flat gray.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
# default output is <input>_motion.mp4 next to the input
uv run motion_extraction.py IMG_6476.MOV

# pick an output path and tune the effect
uv run motion_extraction.py IMG_6476.MOV out.mp4 --delay 2 --alpha 0.5
```

Options:

| flag            | default | meaning                                                        |
| --------------- | ------- | -------------------------------------------------------------- |
| `-d`, `--delay` | `1`     | frame offset between original and inverted copy (`0` = gray)   |
| `-a`, `--alpha` | `0.5`   | opacity of the inverted overlay (0..1)                         |
| `--fourcc`      | `mp4v`  | output codec FourCC                                            |

## Notes

- OpenCV does not copy the audio track; the output is video-only.
- A larger `--delay` produces longer motion trails.
