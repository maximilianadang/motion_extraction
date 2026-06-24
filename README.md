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

Frames are decoded with OpenCV and encoded as **H.264 (yuv420p)** by FFmpeg, so
the output plays anywhere Chromium does — browsers and VS Code video-preview
extensions. The FFmpeg binary is the static one bundled with `imageio-ffmpeg`,
so no system FFmpeg (and no `sudo`) is needed.

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

| flag             | default  | meaning                                                       |
| ---------------- | -------- | ------------------------------------------------------------- |
| `-d`, `--delay`  | `1`      | frame offset between original and inverted copy (`0` = gray)  |
| `-a`, `--alpha`  | `0.5`    | opacity of the inverted overlay (0..1)                        |
| `--crf`          | `18`     | H.264 quality, `0`=lossless .. `51`=worst                     |
| `--preset`       | `medium` | x264 speed/efficiency preset                                  |

## Watching the result in VS Code

VS Code can't play video in a normal editor tab, so use one of:

- **A video-preview extension** — install one from the Marketplace (search
  "mp4" / "Video Preview") and open the `*_motion.mp4` file. The output is H.264,
  which these extensions can decode.
- **Live Preview** (Microsoft extension) — open `preview.html` and click
  *Show Preview*. It serves the folder over `localhost` so the embedded `<video>`
  plays. Edit the `src` in `preview.html` if your output has a different name.

## Notes

- OpenCV does not copy the audio track; the output is video-only.
- A larger `--delay` produces longer motion trails.

## Volumetric plume extraction

`volumetric_extraction.py` is a separate, CPU-only tool that turns a single
"in the wild" plume video into a **3D volumetric prior** (centerline + σ(s) +
relative concentration, plus a voxel grid) aligned with the simulated-lidar /
Gaussian-plume dispersion model in `adrian_experiment/Earth_Field_OpenArea_Model.py`.

```bash
uv run volumetric_extraction.py IMG_6476.MOV --source-xy 0.34 0.55 --source-width-m 1.0
```

Outputs land in `out/volumetric/` (`volume_density.npy`, `plume_profile.csv`,
`volume_meta.json`, previews). See `docs/volumetric-pipeline.md` for the method,
output schema, how it feeds the dispersion model, and its assumptions/limits.
