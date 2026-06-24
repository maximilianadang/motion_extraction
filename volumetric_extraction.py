"""Single-view volumetric plume extraction — CPU-only, "in the wild" video → 3D prior.

Goal: from a single handheld RGB video of a buoyant plume (e.g. smoke from a
chimney), produce a *physically-structured* volumetric prior that a simulated
lidar / Gaussian-plume dispersion pipeline can consume — specifically the kind
in ``adrian_experiment/Earth_Field_OpenArea_Model.py``, which represents a plume
as a centerline trajectory with cross-stream spread ``sigma(s)`` and a
centerline concentration profile.

What this is and isn't
----------------------
This runs on CPU (no GPU) using only numpy + OpenCV. It does NOT attempt the
GPU-heavy neural reconstructions (WildSmoke / Global-Transport / NeRF). Instead
it does classical *single-view axisymmetric plume tomography*:

  1. Temporal-median background → mean smoke OPACITY image (steady plume).
  2. Segment the plume; find its principal axis (PCA).
  3. March along the axis; at each station measure the transverse profile's
     centroid (centerline), second moment (sigma), and line integral L(s).
  4. Assume the cross-section is an axisymmetric Gaussian (this IS the dispersion
     model's own assumption). For that case the measured image opacity is the
     Abel projection of the 3D field in closed form, so the centerline peak
     extinction is  rho_peak(s) = L(s) / (2*pi*sigma(s)^2).
  5. Revolve to a 3D voxel grid of RELATIVE extinction/number-density.

Outputs (relative density; multiply by the model's peak n0 to get part/cm^3):
  * volume_density.npy   — float32 (n_s, n_a, n_b), plume-centric grid
  * volume_meta.json     — voxel sizes (m), axes, scale + assumptions, how to calibrate
  * plume_profile.csv    — s_m, centerline_downwind_x_m, centerline_height_m,
                           sigma_m, rel_peak_density, rel_line_column  (model schema)
  * preview_*.png        — opacity + centerline, sigma(s), a mid-plume slice

The output is a PRIOR on plume SHAPE (geometry + relative concentration). It is
not radiometrically calibrated density; the physics model supplies the absolute
scale. Key assumptions: static background, camera roughly perpendicular to the
wind, axisymmetric Gaussian cross-section. All are recorded in volume_meta.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------
# Stage 1 — video → mean smoke opacity
# ----------------------------------------------------------------------------
def sample_frames(path: Path, max_frames: int, downscale: float) -> tuple[np.ndarray, float]:
    """Return (frames [N,H,W] float32 grayscale, fps) for ~max_frames even samples."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"error: could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total > 0:
        want = set(np.linspace(0, total - 1, min(max_frames, total)).astype(int).tolist())
    else:
        want = None  # keep everything up to max_frames

    frames: list[np.ndarray] = []
    kept: list[int] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if want is None or i in want:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if downscale != 1.0:
                gray = cv2.resize(gray, None, fx=downscale, fy=downscale,
                                  interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32))
            kept.append(i)
            if want is None and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise SystemExit("error: no frames read from the video.")
    return np.stack(frames), fps, np.array(kept)


def stabilize(frames: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove handheld translation by phase-correlating every frame to a mid
    reference. Returns (stabilized frames, mean shift magnitude in px).

    Translation-only is enough to kill the dominant handheld shimmer that
    otherwise makes |frame - background| fire on every static edge.
    """
    ref = frames[len(frames) // 2].astype(np.float32)
    h, w = ref.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    out = np.empty_like(frames)
    mags = []
    for i in range(frames.shape[0]):
        f = frames[i].astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ref, f, win)
        mags.append(math.hypot(dx, dy))
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out[i] = cv2.warpAffine(frames[i], M, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, float(np.mean(mags))


def _longest_true_run(flags: np.ndarray) -> tuple[int, int]:
    """Return (start, end) inclusive of the longest contiguous True run."""
    best_len, best = 0, (0, len(flags) - 1)
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(flags) - 1):
            end = i if (v and i == len(flags) - 1) else i - 1
            if end - start + 1 > best_len:
                best_len, best = end - start + 1, (start, end)
            start = None
    return best


def plume_opacity(frames: np.ndarray, bright: bool = True):
    """Steady mean plume opacity, isolated to the actual release.

    The clip is mostly *setup* (people, equipment moving), with the release a
    single continuous event. Raw brightness-excess can't tell setup motion from
    smoke, so instead we:

    1. Score each frame by mean positive brightness-excess vs a rough median.
    2. Take the **longest contiguous run** of high-score frames = the release
       window (excludes scattered setup outliers).
    3. Rebuild the background from **pre-release frames** (smoke-free, so static
       equipment cancels), and average the smoke signal over the release only.

    ``bright=False`` uses |diff| for dark/sooty plumes.
    Returns (opacity[0..1], background, n_release_frames, n_total).
    """
    rough_bg = np.median(frames, axis=0)
    resid = frames - rough_bg[None]
    sig0 = np.clip(resid, 0.0, None) if bright else np.abs(resid)
    score = sig0.reshape(sig0.shape[0], -1).mean(axis=1)

    lo, hi = float(score.min()), float(score.max())
    above = score >= (lo + 0.35 * (hi - lo))
    if not above.any():
        above = score >= np.percentile(score, 70.0)
    t0, t1 = _longest_true_run(above)

    # Clean background from frames before the release (smoke-free).
    bg = np.median(frames[:t0], axis=0) if t0 >= 5 else rough_bg

    win = frames[t0:t1 + 1] - bg[None]
    sig = np.clip(win, 0.0, None) if bright else np.abs(win)
    mean_sig = sig.mean(axis=0)
    norm = np.percentile(mean_sig, 99.5)
    opacity = np.clip(mean_sig / max(norm, 1e-6), 0.0, 1.0)
    return opacity, bg, int(t1 - t0 + 1), int(frames.shape[0]), int(t0), int(t1)


# ----------------------------------------------------------------------------
# Stage 2 — segment plume + principal axis
# ----------------------------------------------------------------------------
def segment_plume(opacity: np.ndarray, rel_floor: float) -> np.ndarray:
    """Otsu threshold (floored), morphological cleanup, largest connected blob."""
    u8 = np.clip(opacity * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(otsu, rel_floor * 255.0)
    bw = (u8 >= thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        raise SystemExit("error: no plume detected — try lowering --rel-floor.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def principal_axis(opacity: np.ndarray, mask: np.ndarray):
    """Opacity-weighted PCA → (origin[x,y], axis_unit[x,y], perp_unit[x,y])."""
    ys, xs = np.nonzero(mask)
    w = opacity[ys, xs]
    wsum = w.sum()
    mx = float((xs * w).sum() / wsum)
    my = float((ys * w).sum() / wsum)
    dx = xs - mx
    dy = ys - my
    cxx = float((dx * dx * w).sum() / wsum)
    cyy = float((dy * dy * w).sum() / wsum)
    cxy = float((dx * dy * w).sum() / wsum)
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    perp = np.array([-axis[1], axis[0]])
    return np.array([mx, my]), axis, perp


# ----------------------------------------------------------------------------
# Stage 3 — transverse profiles along the axis
# ----------------------------------------------------------------------------
def extract_profiles(opacity, mask, origin, axis, perp, n_stations, source_uv=None):
    """March along the principal axis; per station measure the transverse
    profile's centroid (centerline), second moment (sigma) and line integral.

    Returns (s_from_source_px, center_perp_px, sigma_px, line_px, img_x, img_y),
    sorted by arclength from the source. img_x/img_y are the centerline points in
    image pixels, computed in the centroid-relative frame so they stay correct.

    Source selection:
      * source_uv given (release point, analysis pixels) → parameterize the
        plume *downwind* from it (the axis side carrying more mass). Correct for
        billowing one-sided releases where the dense root is interior.
      * else → narrow-sigma end heuristic (ok for a clean dispersing plume).
    """
    ys, xs = np.nonzero(mask)
    w = opacity[ys, xs]
    px = xs - origin[0]
    py = ys - origin[1]
    s = axis[0] * px + axis[1] * py        # centroid-relative along-axis coord (px)
    t = perp[0] * px + perp[1] * py        # transverse coordinate (px)

    edges = np.linspace(s.min(), s.max(), n_stations + 1)
    ds_px = (edges[-1] - edges[0]) / n_stations
    s_cen, c_arr, sig_arr, line_arr = [], [], [], []
    for i in range(n_stations):
        sel = (s >= edges[i]) & (s < edges[i + 1])
        wi = w[sel]
        if wi.sum() < 1e-3 or sel.sum() < 5:
            continue
        ti = t[sel]
        wsum = wi.sum()
        c = float((ti * wi).sum() / wsum)
        var = float(((ti - c) ** 2 * wi).sum() / wsum)
        s_cen.append(0.5 * (edges[i] + edges[i + 1]))
        c_arr.append(c)
        sig_arr.append(math.sqrt(max(var, 1e-9)))
        line_arr.append(wsum / ds_px)      # ∫ opacity dt per unit along-axis length

    s_cen = np.array(s_cen)
    c_arr = np.array(c_arr)
    sig_arr = np.array(sig_arr)
    line_arr = np.array(line_arr)

    # Correct image-space centerline points (centroid-relative frame).
    img_x = origin[0] + s_cen * axis[0] + c_arr * perp[0]
    img_y = origin[1] + s_cen * axis[1] + c_arr * perp[1]

    if source_uv is not None:
        # Project the release point onto the axis; keep the downwind (more-mass) side.
        s_src = axis[0] * (source_uv[0] - origin[0]) + axis[1] * (source_uv[1] - origin[1])
        mass_pos = line_arr[s_cen > s_src].sum()
        mass_neg = line_arr[s_cen < s_src].sum()
        side = 1.0 if mass_pos >= mass_neg else -1.0
        keep = (np.sign(s_cen - s_src) == side) | np.isclose(s_cen, s_src, atol=1e-6)
        s_cen, c_arr, sig_arr, line_arr, img_x, img_y = (
            a[keep] for a in (s_cen, c_arr, sig_arr, line_arr, img_x, img_y))
        s_from = np.abs(s_cen - s_src)
    else:
        # Narrow-sigma end heuristic; arclength measured from there.
        k = max(3, len(sig_arr) // 5)
        source_s = s_cen[0] if sig_arr[:k].mean() <= sig_arr[-k:].mean() else s_cen[-1]
        s_from = np.abs(s_cen - source_s)

    order = np.argsort(s_from)
    return (s_from[order], c_arr[order], sig_arr[order], line_arr[order],
            img_x[order], img_y[order])


# ----------------------------------------------------------------------------
# Stage 4/5 — closed-form axisymmetric lift → 3D voxel grid
# ----------------------------------------------------------------------------
def build_volume(s_px, sigma_px, line_px, scale_m_per_px, n_perp, sigma_smooth):
    """Revolve the axisymmetric Gaussian cross-section into a relative-density grid.

    For an axisymmetric Gaussian of peak rho_peak and width sigma, the column
    (line) integral of the image opacity satisfies L = rho_peak * 2*pi*sigma^2,
    hence rho_peak(s) = L(s) / (2*pi*sigma^2).  We then place, at each station,
    rho(s,a,b) = rho_peak(s) * exp(-(a^2 + b^2) / (2 sigma^2)).
    """
    s_m = s_px * scale_m_per_px
    sigma_m = np.maximum(sigma_px * scale_m_per_px, 1e-6)
    if sigma_smooth > 1:                    # light smoothing of the sigma profile
        kern = np.ones(sigma_smooth) / sigma_smooth
        sigma_m = np.convolve(sigma_m, kern, mode="same")
    rho_peak = line_px / (2.0 * math.pi * np.maximum(sigma_px, 1e-6) ** 2)  # relative

    radius_m = float(3.0 * sigma_m.max())
    a = np.linspace(-radius_m, radius_m, n_perp)      # crosswind / depth (m)
    b = np.linspace(-radius_m, radius_m, n_perp)      # in-plane transverse (m)
    A, B = np.meshgrid(a, b, indexing="ij")
    rr = A * A + B * B

    vol = np.empty((len(s_m), n_perp, n_perp), dtype=np.float32)
    for i in range(len(s_m)):
        vol[i] = (rho_peak[i] * np.exp(-rr / (2.0 * sigma_m[i] ** 2))).astype(np.float32)

    peak = float(vol.max())
    if peak > 0:
        vol /= peak                          # normalize relative density to 1
    ds_m = float(np.gradient(s_m).mean()) if len(s_m) > 1 else scale_m_per_px
    da_m = float(a[1] - a[0]) if n_perp > 1 else radius_m
    return vol, s_m, sigma_m, rho_peak, peak, (ds_m, da_m, da_m), radius_m


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def centerline_world(img_x, img_y, source_idx, scale):
    """Map the image-space centerline to world (downwind x, height) in metres,
    origin at the source. +x is horizontal away from source; +z is up."""
    src_x, src_y = img_x[source_idx], img_y[source_idx]
    downwind_x = (img_x - src_x) * scale
    height = (src_y - img_y) * scale            # image y grows downward
    return downwind_x, height


def write_outputs(out_dir, vol, voxel_m, radius_m, s_m, sigma_m, rho_peak, peak_rel,
                  downwind_x, height, line_px, scale, meta_extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "volume_density.npy", vol)

    rel_line = line_px / max(float(line_px.max()), 1e-9)
    rel_peak = rho_peak / max(float(rho_peak.max()), 1e-9)
    with (out_dir / "plume_profile.csv").open("w", newline="") as fh:
        fh.write("s_m,centerline_downwind_x_m,centerline_height_m,sigma_m,"
                 "rel_peak_density,rel_line_column\n")
        for i in range(len(s_m)):
            fh.write(f"{s_m[i]:.4f},{downwind_x[i]:.4f},{height[i]:.4f},"
                     f"{sigma_m[i]:.4f},{rel_peak[i]:.6f},{rel_line[i]:.6f}\n")

    meta = {
        "description": "Relative volumetric extinction/number-density prior from a "
                       "single-view plume video. Multiply by the dispersion model's "
                       "peak concentration n0 (part/cm^3) to calibrate.",
        "array_file": "volume_density.npy",
        "array_shape": list(vol.shape),
        "axes": {
            "0": "s — along-plume arclength (source at index 0)",
            "1": "a — crosswind / line-of-sight depth (axisymmetric)",
            "2": "b — in-plane transverse (vertical-ish)",
        },
        "voxel_size_m": {"ds": voxel_m[0], "da": voxel_m[1], "db": voxel_m[2]},
        "cross_section_radius_m": radius_m,
        "scale_m_per_px": scale,
        "units": "relative (0..1); not radiometrically calibrated",
        "calibrate_to_part_per_cm3": "n(s,a,b) = volume_density * n0_peak",
        "assumptions": [
            "static background (temporal-median subtraction)",
            "camera roughly perpendicular to wind",
            "axisymmetric Gaussian cross-section (matches Gaussian-plume model)",
            "time-averaged steady plume",
        ],
        "consumes_into": "adrian_experiment Gaussian-plume / lidar model: use "
                         "plume_profile.csv as measured centerline + sigma(s) + "
                         "relative concentration, or ray-march volume_density.npy.",
    }
    meta.update(meta_extra)
    (out_dir / "volume_meta.json").write_text(json.dumps(meta, indent=2))


def write_overlay(out_dir, color_frame, analysis_hw, mask, img_x, img_y, sigma_px, perp):
    """Draw mask outline + centerline + ±2σ + source/tip on a real release frame."""
    h, w = color_frame.shape[:2]
    ah, aw = analysis_hw
    sx, sy = w / aw, h / ah
    img = color_frame.copy()
    cnts, _ = cv2.findContours((mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        cc = cnt.reshape(-1, 2).astype(np.float64)
        cc[:, 0] *= sx; cc[:, 1] *= sy
        cv2.polylines(img, [cc.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    def sc(xs, ys):
        return np.stack([xs * sx, ys * sy], 1).astype(np.int32)

    cv2.polylines(img, [sc(img_x + 2 * sigma_px * perp[0], img_y + 2 * sigma_px * perp[1])],
                  False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [sc(img_x - 2 * sigma_px * perp[0], img_y - 2 * sigma_px * perp[1])],
                  False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [sc(img_x, img_y)], False, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(sc(img_x[:1], img_y[:1])[0]), 12, (0, 255, 0), -1)
    cv2.circle(img, tuple(sc(img_x[-1:], img_y[-1:])[0]), 12, (255, 0, 255), -1)
    cv2.imwrite(str(out_dir / "preview_overlay_frame.png"), img)


def write_previews(out_dir, opacity, mask, img_x, img_y, sigma_px, perp,
                   s_m, sigma_m, vol):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping previews: matplotlib unavailable: {exc})")
        return

    env_x1 = img_x + 2 * sigma_px * perp[0]
    env_y1 = img_y + 2 * sigma_px * perp[1]
    env_x2 = img_x - 2 * sigma_px * perp[0]
    env_y2 = img_y - 2 * sigma_px * perp[1]
    plt.figure(figsize=(8, 4.5))
    plt.imshow(opacity, cmap="inferno")
    plt.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
    plt.plot(env_x1, env_y1, "c-", lw=0.8)
    plt.plot(env_x2, env_y2, "c-", lw=0.8, label="±2σ envelope")
    plt.plot(img_x, img_y, "w-", lw=1.4, label="centerline")
    plt.plot(img_x[0], img_y[0], "go", ms=7, label="source")
    plt.title("Mean plume opacity + extracted centerline / spread")
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "preview_opacity_centerline.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(s_m, sigma_m)
    plt.xlabel("along-plume arclength s [m]")
    plt.ylabel("sigma(s) [m]")
    plt.title("Measured plume spread")
    plt.tight_layout()
    plt.savefig(out_dir / "preview_sigma_profile.png", dpi=150)
    plt.close()

    mid = vol.shape[0] // 2
    plt.figure()
    plt.imshow(vol[mid], cmap="viridis", origin="lower")
    plt.title(f"Cross-section slice at s index {mid}")
    plt.xlabel("b (transverse)")
    plt.ylabel("a (depth)")
    plt.tight_layout()
    plt.savefig(out_dir / "preview_cross_section.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
def run(args):
    print(f"[1/5] sampling frames from {args.input} ...")
    frames, fps, idxs = sample_frames(args.input, args.max_frames, args.downscale)
    print(f"      {frames.shape[0]} frames @ {frames.shape[2]}x{frames.shape[1]} (fps~{fps:.1f})")

    if args.stabilize:
        frames, shake = stabilize(frames)
        print(f"      stabilized (mean shift {shake:.1f} px)")

    print("[2/5] plume opacity (release-windowed) + segmentation ...")
    opacity, _, n_active, n_total, t0, t1 = plume_opacity(frames, bright=not args.dark_plume)
    print(f"      release window: {n_active}/{n_total} frames "
          f"(orig {int(idxs[t0])}-{int(idxs[t1])})")
    mask = segment_plume(opacity, args.rel_floor)
    origin, axis, perp = principal_axis(opacity, mask)
    print(f"      plume axis (img) = ({axis[0]:+.2f}, {axis[1]:+.2f}); "
          f"{int(mask.sum())} px")

    print("[3/5] transverse profiles ...")
    source_uv = None
    if args.source_xy is not None:
        source_uv = (args.source_xy[0] * frames.shape[2], args.source_xy[1] * frames.shape[1])
        print(f"      release point fixed at u,v = {args.source_xy} "
              f"(px {source_uv[0]:.0f},{source_uv[1]:.0f})")
    else:
        print("      release point: narrow-end heuristic (pass --source-xy for a controlled release)")
    s_px, c_px, sigma_px, line_px, img_x, img_y = extract_profiles(
        opacity, mask, origin, axis, perp, args.stations, source_uv)
    if len(s_px) < 4:
        raise SystemExit("error: too few valid stations; try --downscale 1 or more frames.")

    # Trim the faint dispersing tail, where the plume falls below detectability
    # and the L/(2*pi*sigma^2) peak estimate becomes unreliable.
    keep = np.where(line_px >= args.min_column_frac * float(line_px.max()))[0]
    if len(keep):
        last = int(keep.max()) + 1
        s_px, c_px, sigma_px, line_px, img_x, img_y = (
            a[:last] for a in (s_px, c_px, sigma_px, line_px, img_x, img_y))

    # scale: explicit m/px preferred; else a rough estimate from the assumed
    # source width (~ visible full width ≈ 4*sigma at the narrowest inner section).
    src_idx = 0
    if args.scale_m_per_px is not None:
        scale = args.scale_m_per_px
        scale_src = "from --scale-m-per-px"
    else:
        inner = sigma_px[: max(3, len(sigma_px) // 10)]
        ref_width_px = max(4.0 * float(np.median(inner)), 1.0)
        scale = args.source_width_m / ref_width_px
        scale_src = "estimated from --source-width-m (approximate; calibrate for real metres)"
    print(f"      {len(s_px)} stations; scale = {scale*100:.3f} cm/px ({scale_src})")

    print("[4/5] axisymmetric lift → voxel grid ...")
    vol, s_m, sigma_m, rho_peak, _, voxel_m, radius_m = build_volume(
        s_px, sigma_px, line_px, scale, args.grid_perp, args.sigma_smooth)
    downwind_x, height = centerline_world(img_x, img_y, src_idx, scale)
    print(f"      volume {vol.shape}; plume length {s_m.max():.2f} m, "
          f"max sigma {sigma_m.max():.2f} m")

    print("[5/5] writing outputs ...")
    write_outputs(args.output, vol, voxel_m, radius_m, s_m, sigma_m, rho_peak,
                  float(vol.max()), downwind_x, height, line_px, scale,
                  {"source_video": str(args.input), "fps": fps,
                   "active_release_frames": f"{n_active}/{n_total}",
                   "scale_provenance": scale_src,
                   "source_width_m_assumed": args.source_width_m
                   if args.scale_m_per_px is None else None,
                   "release_point_uv": args.source_xy})
    if not args.no_preview:
        write_previews(args.output, opacity, mask, img_x, img_y, sigma_px, perp,
                       s_m, sigma_m, vol)
        cap = cv2.VideoCapture(str(args.input))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idxs[(t0 + t1) // 2]))
        ok, color = cap.read()
        cap.release()
        if ok:
            write_overlay(args.output, color, opacity.shape, mask,
                          img_x, img_y, sigma_px, perp)
        try:
            import render_volume
            render_volume.main([str(args.output)])
        except Exception as exc:  # pragma: no cover - rendering is optional
            print(f"(volume render skipped: {exc})")
    print(f"\nDone. Outputs in {args.output}/")
    print(f"  plume length      : {s_m.max():.2f} m")
    print(f"  sigma range       : {sigma_m.min():.2f} – {sigma_m.max():.2f} m")
    print(f"  centerline rise   : {height.min():.2f} – {height.max():.2f} m")
    print("  files             : volume_density.npy, volume_meta.json, "
          "plume_profile.csv, preview_*.png")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="input plume video")
    p.add_argument("-o", "--output", type=Path, default=Path("out/volumetric"),
                   help="output directory (default: out/volumetric)")
    p.add_argument("--max-frames", type=int, default=150, help="frames to sample (default 150)")
    p.add_argument("--downscale", type=float, default=0.5,
                   help="analysis downscale factor, 0<f<=1 (default 0.5)")
    p.add_argument("--stations", type=int, default=160,
                   help="along-plume sampling stations (default 160)")
    p.add_argument("--source-xy", type=float, nargs=2, default=None, metavar=("U", "V"),
                   help="release point in normalized image coords 0..1 (e.g. 0.35 0.54); "
                        "recommended for a controlled release. Else a shape heuristic is used.")
    p.add_argument("--grid-perp", type=int, default=96,
                   help="cross-section grid resolution per side (default 96)")
    p.add_argument("--rel-floor", type=float, default=0.08,
                   help="minimum opacity fraction for segmentation (default 0.08)")
    p.add_argument("--sigma-smooth", type=int, default=5,
                   help="boxcar window for smoothing sigma(s) (default 5)")
    p.add_argument("--min-column-frac", type=float, default=0.05,
                   help="trim the dispersing tail below this fraction of peak column "
                        "density (default 0.05)")
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--source-width-m", type=float, default=1.0,
                       help="assumed plume width at the source [m] for pixel scaling (default 1.0)")
    scale.add_argument("--scale-m-per-px", type=float, default=None,
                       help="explicit metres-per-pixel (overrides --source-width-m)")
    p.add_argument("--stabilize", action="store_true",
                   help="enable phase-correlation stabilization (only for genuinely "
                        "handheld footage; off by default — a static camera is better left alone)")
    p.add_argument("--dark-plume", action="store_true",
                   help="treat the plume as darker than the background (sooty smoke) "
                        "instead of bright/white scattering smoke")
    p.add_argument("--no-preview", action="store_true", help="skip preview PNGs")
    args = p.parse_args(argv)
    if not args.input.exists():
        p.error(f"input does not exist: {args.input}")
    if not 0.0 < args.downscale <= 1.0:
        p.error("--downscale must be in (0, 1]")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
