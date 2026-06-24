# Volumetric plume extraction pipeline

`volumetric_extraction.py` turns a single "in the wild" RGB video of a buoyant
plume (e.g. the vineyard aerosol-release clip `IMG_6476.MOV`) into a
**volumetric prior** for the simulated-lidar / Gaussian-plume dispersion model in
`adrian_experiment/Earth_Field_OpenArea_Model.py`.

It is **CPU-only** (numpy + OpenCV, no GPU) and runs in a few seconds on this
machine — the GPU-heavy neural reconstructions (WildSmoke / Global-Transport /
NeRF, see `cloud-volume-from-video.md`) are deliberately avoided as not
onboard-feasible. Instead it does classical **single-view axisymmetric plume
tomography**, which maps directly onto the dispersion model's own Gaussian
representation.

## Why this is aligned with the lidar model

`Earth_Field_OpenArea_Model.py` represents a plume as a **centerline trajectory**,
cross-stream spreads **σ(s)**, and a **centerline concentration** profile; the
end-to-end lidar then integrates backscatter β and extinction σ along the beam
through that field. So the most useful thing a video can supply is the plume's
**geometry + relative concentration** (the *shape*), with the physics model
supplying the absolute magnitude (peak `n0`). This pipeline outputs exactly that.

## Method (5 stages)

1. **Windowed background subtraction.** Sample ~150 frames; background = temporal
   median; per-pixel smoke signal = positive brightness excess (`relu(frame−bg)`,
   for bright/scattering smoke). Auto-select the **active-release frames** so the
   setup period (people, equipment) doesn't pollute the mean.
2. **Segment + principal axis.** Otsu threshold → largest blob → opacity-weighted
   PCA gives the plume axis.
3. **Transverse profiles.** March along the axis; per station measure the
   profile's centroid (centerline), second moment (σ), and line integral L(s).
4. **Axisymmetric Gaussian lift (closed form).** For an axisymmetric Gaussian
   cross-section the image opacity is the Abel projection in closed form, so the
   centerline peak extinction is `ρ_peak(s) = L(s) / (2π·σ(s)²)` — no noisy
   numerical inversion.
5. **Voxelize.** Revolve the cross-section into a 3-D relative-density grid.

### "In the wild" handling
- **Auto-windowing** to the active release (the clip is mostly setup).
- **Static-camera assumption** (this footage is tripod-steady); phase-correlation
  stabilization is opt-in (`--stabilize`) because it *hurt* on dissimilar
  setup-vs-release frames.
- **Source anchoring** (`--source-xy U V`): a billowing release has its dense root
  *interior*, so the nozzle can't be guessed from shape — anchor it and the plume
  is parameterized downwind (the axis side carrying more mass).

## Outputs (in `out/volumetric/`)

| file | contents |
|---|---|
| `volume_density.npy` | float32 `(n_s, n_a, n_b)` relative extinction/number-density, plume-centric (s along-plume, a depth, b transverse) |
| `volume_meta.json` | voxel sizes (m), axes, scale + assumptions, how to calibrate |
| `plume_profile.csv` | `s_m, centerline_downwind_x_m, centerline_height_m, sigma_m, rel_peak_density, rel_line_column` — the dispersion model's schema |
| `preview_*.png` | opacity + centerline/±2σ envelope, σ(s), a cross-section slice |
| `render_volume.png` | 6-panel: 3D world-placed plume, orthogonal max-projections, end-on cross-section, re-projected silhouette (validation), centerline+spread |
| `render_volume_3d.png` | dedicated 3D view of the reconstructed volume from 3 angles |

The renders are produced automatically at the end of a run, or standalone via
`uv run render_volume.py out/volumetric`.

The density is **relative (0..1)**; multiply by the model's peak `n0` (part/cm³) to
calibrate: `n(s,a,b) = volume_density · n0_peak`.

## Usage

```bash
# controlled release: anchor the nozzle (normalized image coords)
uv run volumetric_extraction.py IMG_6476.MOV --source-xy 0.34 0.55 --source-width-m 1.0

# if you know the real scale from a reference object, set it directly
uv run volumetric_extraction.py IMG_6476.MOV --source-xy 0.34 0.55 --scale-m-per-px 0.035
```
Key flags: `--source-xy` (release point), `--scale-m-per-px` / `--source-width-m`
(scale), `--dark-plume` (sooty), `--stabilize` (handheld), `--min-column-frac`
(tail trim), `--downscale`, `--max-frames`, `--stations`, `--grid-perp`.

## Feeding the dispersion / lidar model

`plume_profile.csv` is the bridge: its `s_m`, `centerline_*`, `sigma_m`, and
`rel_*` columns are the *measured* analogue of the model's `compute_near_field`
arrays (`arclength_m`, `height_m`, σ, `cm`). Use it to replace/constrain the
analytic Smith–Mungal / Briggs spreads with measured ones, or ray-march
`volume_density.npy` directly for backscatter/extinction integrals.

## Assumptions & limitations (honest scope)

- **Relative, not calibrated.** Output is a shape/relative-density prior, not
  kg/m³ or part/cm³ — the physics model supplies absolute magnitude.
- **Scale needs a reference.** m/px is estimated from an assumed source width
  (flagged approximate in the metadata); set `--scale-m-per-px` from a known
  object for real metres.
- **Axisymmetric Gaussian cross-section** (the model's own assumption); the
  crosswind depth σ_y is taken ≈ the measured in-plane σ.
- **Single static view**; camera ≈ perpendicular to the wind.
- **Dispersing-tail noise.** Where the plume thins below threshold, the detected
  σ narrows and `ρ_peak = L/2πσ²` can spike; `--min-column-frac` trims the worst
  of it, but trust the near/mid field most.
