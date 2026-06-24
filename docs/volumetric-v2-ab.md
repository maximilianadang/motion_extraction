# Volumetric v2 — A/B study of physics improvements

`volumetric_v2.py` is a **nondestructive** parallel build (the original
`volumetric_extraction.py` is unchanged). It adds a hard validation —
**re-project the reconstructed 3D volume back onto the RGB frame** — and A/B-tests
methodology changes one at a time, scoring each against the *same* fixed
reference (the raw observed brightness-excess smoke).

## The validation: re-projection onto the frame
`reproject_volume()` integrates the plume-centric volume through its depth axis
and rasterizes it back along the bent centerline into image space; `overlay_on_frame()`
composites it on the real release frame. If the projected density doesn't land on
the smoke pixels, the method is wrong. All methods below pass this geometric
sanity check (source at nozzle, plume downwind); they differ in *quality*.

## Metrics
- **corr / IoU** vs the raw observed smoke — data fit (↑ better), but biased: it
  rewards reproducing the noisy mean appearance, which the baseline does almost by
  construction, and penalizes physical regularization that deviates from it.
- **rough** = mean |Δρ_peak| / mean ρ_peak along the plume — unphysical banding
  (↓ better). Added precisely because corr alone can't see "physical plausibility."

## Results (IMG_6476.MOV)

| method | corr ↑ | IoU ↑ | rough ↓ | verdict |
|---|---|---|---|---|
| **baseline** | 0.729 | 0.251 | 0.091 | reference |
| **smooth** (steady-plume) | 0.725 | **0.254** | **0.062** | **adopted** |
| continuity (flow mass-cons.) | 0.646 | 0.226 | 0.103 | rejected |
| beer_lambert (optical depth) | 0.577 | 0.041 | 0.075 | rejected |

## What we learned (intuition + overlays, not just numbers)

- **Steady-plume smoothing wins.** Smoothing the column-density and width profiles
  cut banding by a third (0.091→0.062) at *no* cost to data fit. A steady mean
  plume genuinely varies smoothly downwind, so this is principled, not cosmetic.
  **Composed into the final deliverable.**
- **Beer–Lambert opacity fails — and the overlay shows why.** It collapsed to a
  thin streak on the *sunlit top edge*: naïve `alpha=(I−bg)/(L_smoke−bg)` confounds
  **illumination with opacity**, so the brightly-lit top saturates to huge τ. A
  correct optical-depth needs an illumination model. Rejected.
- **Optical-flow mass-continuity doesn't help here.** The flow *direction* is
  reliable (arrows follow the plume), but the **magnitude is too noisy** on the
  textureless smoke interior to drive a u·L=const constraint. Rejected as a
  density regularizer — but the **velocity field is kept as a transport product**
  (`velocity_profile.csv`), which the dispersion/lidar model wants anyway.

## The honest limit (unchanged)
Transport did **not** add depth information: the plume advects roughly *in the
image plane* (perpendicular to the camera), so successive frames reveal no new
viewing angles. The recovered volume is still a single-view **axisymmetric** shape
prior — now with smoother, more physical density and a measured velocity field.
True depth still requires angular diversity (a second camera) or a full
transport-tomography optimization over the time sequence (GPU-scale).

## Deliverable
`out/volumetric_v2/`: smoothed `volume_density.npy`, `plume_profile.csv`,
`velocity_profile.csv`, `reproject_overlay.png` (the sanity check on the frame),
and `render_volume*.png`. Regenerate with:

```bash
uv run volumetric_v2.py IMG_6476.MOV --finalize     # adopted method
uv run volumetric_v2.py IMG_6476.MOV --method all   # re-run the A/B
```
