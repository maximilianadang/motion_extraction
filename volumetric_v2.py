"""Improved single-view plume volume recovery — A/B harness (nondestructive).

This sits alongside `volumetric_extraction.py` (unchanged) and lets us test
methodology changes one at a time against a hard validation: **re-project the
reconstructed 3D volume back onto the RGB frame it came from**. If the projected
density doesn't land on the smoke pixels, the method is wrong — full stop.

Methods (composed incrementally once each is shown to help):
  baseline      — mean brightness-excess opacity, axisymmetric Gaussian revolve
  beer_lambert  — physical optical-depth (airlight model) instead of raw excess
  flow          — optical-flow transport: centerline = advection streamline,
                  plus a measured velocity field for the lidar prior
  continuity    — steady-plume mass conservation regularizes concentration

Each run produces a re-projection overlay on the real frame and match metrics
(correlation + IoU vs the measured opacity), so improvements are A/B-comparable.
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np

import volumetric_extraction as ve


# ----------------------------------------------------------------------------
# Shared front-end (reuses the validated extraction stages)
# ----------------------------------------------------------------------------
def opacity_beer_lambert(frames, t0, t1, bg):
    """Physical optical depth from a scattering/airlight model, instead of raw
    brightness excess. White smoke composites over the background as
        I = bg·T + L_smoke·(1−T),   alpha = 1−T = (I−bg)/(L_smoke−bg)
    so the column optical depth is τ = −ln(T) = −ln(1−alpha). This de-saturates
    thick smoke (where raw excess clips) → column density closer to actual mass.
    """
    win = frames[t0:t1 + 1]
    l_smoke = float(np.percentile(win, 99.5))           # bright sunlit smoke
    denom = max(l_smoke - float(np.median(bg)), 1.0)
    alpha = np.clip((win - bg[None]) / denom, 0.0, 0.98)
    tau = -np.log(1.0 - alpha)
    mean_tau = tau.mean(axis=0)
    return np.clip(mean_tau / max(np.percentile(mean_tau, 99.5), 1e-6), 0.0, 1.0)


def extract_frontend(video, max_frames=150, downscale=0.5,
                     source_xy=(0.34, 0.55), source_width_m=1.0, stations=160,
                     opacity_mode="excess"):
    frames, fps, idxs = ve.sample_frames(pathlib.Path(video), max_frames, downscale)
    op_excess, bg, nrel, ntot, t0, t1 = ve.plume_opacity(frames, bright=True)
    op = (opacity_beer_lambert(frames, t0, t1, bg)
          if opacity_mode == "beer_lambert" else op_excess)
    mask = ve.segment_plume(op, 0.08)
    origin, axis, perp = ve.principal_axis(op, mask)
    suv = (source_xy[0] * op.shape[1], source_xy[1] * op.shape[0])
    s, c, sig, line, ix, iy = ve.extract_profiles(op, mask, origin, axis, perp, stations, suv)
    keep = np.where(line >= 0.05 * line.max())[0]
    last = int(keep.max()) + 1
    s, c, sig, line, ix, iy = (a[:last] for a in (s, c, sig, line, ix, iy))
    inner = sig[: max(3, len(sig) // 10)]
    scale = source_width_m / max(4.0 * float(np.median(inner)), 1.0)
    return dict(frames=frames, idxs=idxs, op=op, op_excess=op_excess, bg=bg,
                mask=mask, origin=origin, axis=axis, perp=perp, s=s, c=c, sig=sig,
                line=line, ix=ix, iy=iy, scale=scale, fps=fps,
                window=(int(t0), int(t1)), ahw=op.shape)


# ----------------------------------------------------------------------------
# Re-projection: 3D volume -> image (orthographic along the depth axis)
# ----------------------------------------------------------------------------
def reproject_volume(vol, ix, iy, perp, scale, radius_m, ahw, upsample=4):
    """Sum the plume-centric volume through depth (axis 1) and rasterize it back
    into image space along the bent centerline, with dense bilinear splatting so
    the projection is a smooth filled silhouette. Returns a (H,W) opacity raster."""
    n_s, n_a, n_b = vol.shape
    proj = vol.sum(axis=1)                           # (n_s, n_b), integrate depth

    # Upsample along the plume (s) and transverse (b) so adjacent splats overlap.
    ns, nb = n_s * upsample, n_b * upsample
    proj_f = cv2.resize(proj, (nb, ns), interpolation=cv2.INTER_LINEAR)
    si = np.linspace(0, n_s - 1, ns)
    ixf = np.interp(si, np.arange(n_s), ix)
    iyf = np.interp(si, np.arange(n_s), iy)
    b_px = np.linspace(-radius_m, radius_m, nb) / scale

    px = ixf[:, None] + b_px[None, :] * perp[0]
    py = iyf[:, None] + b_px[None, :] * perp[1]
    h, w = ahw
    raster = np.zeros((h, w), np.float32)
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = x0 + dx, y0 + dy
            wgt = (1 - np.abs(px - xx)) * (1 - np.abs(py - yy))
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            np.add.at(raster, (yy[ok], xx[ok]), (proj_f * wgt)[ok])
    return cv2.GaussianBlur(raster, (0, 0), 1.5)


def match_metrics(raster, measured_op):
    a = raster / max(float(raster.max()), 1e-9)
    b = measured_op / max(float(measured_op.max()), 1e-9)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    am, bm = a > 0.1, b > 0.1
    iou = float((am & bm).sum() / max((am | bm).sum(), 1))
    return corr, iou


def overlay_on_frame(raster, color_frame, out_path, label=""):
    rn = raster / max(float(raster.max()), 1e-9)
    big = cv2.resize(rn, (color_frame.shape[1], color_frame.shape[0]))
    heat = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(big * 1.3, 0, 0.65)[..., None]
    blend = (color_frame * (1 - alpha) + heat * alpha).astype(np.uint8)
    if label:
        cv2.putText(blend, label, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(out_path), blend)


# ----------------------------------------------------------------------------
# Transport: optical-flow velocity field over the release window
# ----------------------------------------------------------------------------
def compute_mean_flow(video, f0, f1, downscale=0.5, max_pairs=40):
    """Mean dense optical flow (px per `step` frames) across the release window."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f0))
    step = max(1, (f1 - f0) // max_pairs)
    prev, acc, n = None, None, 0
    fi = int(f0)
    while fi <= f1:
        ok, fr = cap.read()
        if not ok:
            break
        if (fi - f0) % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if downscale != 1.0:
                g = cv2.resize(g, None, fx=downscale, fy=downscale,
                               interpolation=cv2.INTER_AREA)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None,
                                                    0.5, 3, 25, 3, 5, 1.2, 0)
                acc = flow if acc is None else acc + flow
                n += 1
            prev = g
        fi += 1
    cap.release()
    return (acc / max(n, 1)).astype(np.float32), step


def velocity_along_centerline(flow, ix, iy, axis, scale, fps, step, smooth=7):
    """Plume-axis speed u(s) in m/s sampled along the image centerline."""
    h, w = flow.shape[:2]
    xi = np.clip(np.round(ix).astype(int), 0, w - 1)
    yi = np.clip(np.round(iy).astype(int), 0, h - 1)
    fx, fy = flow[yi, xi, 0], flow[yi, xi, 1]
    along_px = fx * axis[0] + fy * axis[1]           # px per `step` frames
    u_mps = np.abs(along_px) * scale * fps / step    # m/s
    if smooth > 1:
        k = np.ones(smooth) / smooth
        u_mps = np.convolve(u_mps, k, mode="same")
    return np.maximum(u_mps, 1e-3)


def draw_flow_quiver(flow, color_frame, mask, scale, fps, step, out_path):
    """Sanity overlay: flow arrows on the frame (should follow the plume)."""
    h, w = flow.shape[:2]
    img = color_frame.copy()
    sx, sy = img.shape[1] / w, img.shape[0] / h
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            if not mask[y, x]:
                continue
            fxp, fyp = flow[y, x]
            spd = np.hypot(fxp, fyp) * scale * fps / step
            p0 = (int(x * sx), int(y * sy))
            p1 = (int((x + fxp * 3) * sx), int((y + fyp * 3) * sy))
            cv2.arrowedLine(img, p0, p1, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)
            _ = spd
    cv2.imwrite(str(out_path), img)


# ----------------------------------------------------------------------------
# Reconstruction methods
# ----------------------------------------------------------------------------
def build_baseline(ctx, video=None, grid_perp=96):
    vol, *_rest, radius_m = ve.build_volume(
        ctx["s"], ctx["sig"], ctx["line"], ctx["scale"], grid_perp, 5)
    return vol, radius_m, {}


def robust_velocity(flow, ix, iy, scale, fps, step, band=6, smooth=9):
    """Plume speed u(s) [m/s] from median flow magnitude in a band around the
    centerline (robust to the textureless interior), floored to avoid blow-up."""
    h, w = flow.shape[:2]
    u = np.zeros(len(ix))
    for i in range(len(ix)):
        x0, y0 = int(ix[i]), int(iy[i])
        xs = slice(max(0, x0 - band), min(w, x0 + band + 1))
        ys = slice(max(0, y0 - band), min(h, y0 + band + 1))
        u[i] = np.hypot(np.median(flow[ys, xs, 0]), np.median(flow[ys, xs, 1]))
    u_mps = u * scale * fps / step
    if smooth > 1:
        u_mps = np.convolve(u_mps, np.ones(smooth) / smooth, mode="same")
    return np.maximum(u_mps, 0.2 * np.percentile(u_mps, 90))


def build_continuity(ctx, video, grid_perp=96):
    """Mass-conservation regularization: a steady plume has constant mass flux
    Φ(s)=u(s)·L(s). Use the measured velocity to pull the (noisy) column density
    toward Φ0/u(s), then smooth → a physically steadier concentration profile."""
    f0 = int(ctx["idxs"][ctx["window"][0]])
    f1 = int(ctx["idxs"][ctx["window"][1]])
    flow, step = compute_mean_flow(video, f0, f1, downscale=0.5)
    u = robust_velocity(flow, ctx["ix"], ctx["iy"], ctx["scale"], ctx["fps"], step)
    L = ctx["line"].astype(float)
    flux0 = float(np.median(u * L))
    L_cont = flux0 / u
    L_new = np.sqrt(np.clip(L, 1e-6, None) * np.clip(L_cont, 1e-6, None))
    L_new = np.convolve(L_new, np.ones(7) / 7, mode="same")
    vol, *_rest, radius_m = ve.build_volume(
        ctx["s"], ctx["sig"], L_new, ctx["scale"], grid_perp, 5)
    return vol, radius_m, {"u_mps": u}


def _smooth(a, w):
    return np.convolve(a, np.ones(w) / w, mode="same") if w > 1 else a


def build_smooth(ctx, video=None, grid_perp=96, win=11):
    """Steady-plume regularization: smooth the column density and width profiles
    (a steady mean plume varies smoothly downwind) to remove the unphysical
    ρ_peak banding — without relying on the noisy flow or shading-confounded τ."""
    L = _smooth(ctx["line"].astype(float), win)
    sig = _smooth(ctx["sig"].astype(float), win)
    vol, *_rest, radius_m = ve.build_volume(ctx["s"], sig, L, ctx["scale"], grid_perp, win)
    return vol, radius_m, {}


# method name -> (opacity_mode, build_fn)
METHODS = {
    "baseline": ("excess", build_baseline),
    "smooth": ("excess", build_smooth),
    "beer_lambert": ("beer_lambert", build_baseline),
    "continuity": ("excess", build_continuity),
    "combined": ("excess", build_smooth),  # excess + steady-plume smoothing
}


def run_method(video, method, out_dir, source_xy=(0.34, 0.55)):
    opacity_mode, build_fn = METHODS[method]
    ctx = extract_frontend(video, source_xy=source_xy, opacity_mode=opacity_mode)
    vol, radius_m, extras = build_fn(ctx, video)
    raster = reproject_volume(vol, ctx["ix"], ctx["iy"], ctx["perp"],
                              ctx["scale"], radius_m, ctx["ahw"])
    # always score against the same fixed reference: the raw observed smoke
    corr, iou = match_metrics(raster, ctx["op_excess"])
    # physical-plausibility proxy: along-plume roughness of the peak density
    peak_s = vol.max(axis=(1, 2))
    rough = float(np.mean(np.abs(np.diff(peak_s))) / max(np.mean(peak_s), 1e-9))

    t0, t1 = ctx["window"]
    fidx = int(ctx["idxs"][(t0 + t1) // 2])
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, col = cap.read()
    cap.release()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ok:
        overlay_on_frame(raster, col, out_dir / f"reproj_{method}.png",
                         label=f"{method}  corr={corr:.3f} IoU={iou:.3f} rough={rough:.3f}")
    print(f"{method:14s} corr={corr:.3f}  IoU={iou:.3f}  rough={rough:.3f}  vol{vol.shape}")
    return dict(method=method, corr=corr, iou=iou, rough=rough, vol=vol,
                ctx=ctx, extras=extras)


def finalize(video, out_dir="out/volumetric_v2", source_xy=(0.34, 0.55)):
    """Produce the improved deliverable: A/B-selected method (robust release-window
    mean + steady-plume smoothing), the re-projection overlay (sanity check), the
    transport velocity field, and the 3D render."""
    ctx = extract_frontend(video, source_xy=source_xy, opacity_mode="excess")
    L = _smooth(ctx["line"].astype(float), 11)
    sig = _smooth(ctx["sig"].astype(float), 11)
    vol, s_m, sigma_m, rho_peak, _peak, voxel_m, radius_m = ve.build_volume(
        ctx["s"], sig, L, ctx["scale"], 96, 11)
    dwx, hgt = ve.centerline_world(ctx["ix"], ctx["iy"], 0, ctx["scale"])

    f0, f1 = int(ctx["idxs"][ctx["window"][0]]), int(ctx["idxs"][ctx["window"][1]])
    flow, step = compute_mean_flow(video, f0, f1, downscale=0.5)
    u = robust_velocity(flow, ctx["ix"], ctx["iy"], ctx["scale"], ctx["fps"], step)

    out = pathlib.Path(out_dir)
    ve.write_outputs(out, vol, voxel_m, radius_m, s_m, sigma_m, rho_peak,
                     float(vol.max()), dwx, hgt, L, ctx["scale"],
                     {"method": "v2 = robust release-window mean + steady-plume "
                                "smoothing (A/B-selected over beer-lambert/continuity)",
                      "source_video": str(video),
                      "transport_velocity": "see velocity_profile.csv (flow direction "
                                            "reliable; magnitude approximate)"})
    with (out / "velocity_profile.csv").open("w") as fh:
        fh.write("s_m,u_mps\n")
        for i in range(len(s_m)):
            fh.write(f"{s_m[i]:.4f},{u[i]:.4f}\n")

    raster = reproject_volume(vol, ctx["ix"], ctx["iy"], ctx["perp"],
                              ctx["scale"], radius_m, ctx["ahw"])
    corr, iou = match_metrics(raster, ctx["op_excess"])
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, (f0 + f1) // 2)
    ok, col = cap.read()
    cap.release()
    if ok:
        overlay_on_frame(raster, col, out / "reproject_overlay.png",
                         f"v2 reprojection  corr={corr:.3f} IoU={iou:.3f}")
    try:
        import render_volume
        render_volume.main([str(out)])
    except Exception as exc:  # pragma: no cover
        print(f"(render skipped: {exc})")
    print(f"finalized -> {out}/  reproj corr={corr:.3f} IoU={iou:.3f}; "
          f"velocity {u.min():.2f}-{u.max():.2f} m/s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="A/B harness for plume volume recovery")
    p.add_argument("input")
    p.add_argument("--method", default="baseline", choices=list(METHODS) + ["all"])
    p.add_argument("--out", default="out/ab")
    p.add_argument("--source-xy", type=float, nargs=2, default=[0.34, 0.55])
    p.add_argument("--finalize", action="store_true",
                   help="write the A/B-selected improved deliverable to out/volumetric_v2")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.finalize:
        finalize(a.input, "out/volumetric_v2", tuple(a.source_xy))
        return
    methods = list(METHODS) if a.method == "all" else [a.method]
    results = [run_method(a.input, m, a.out, tuple(a.source_xy)) for m in methods]
    if len(results) > 1:
        print("\n=== A/B summary (corr/IoU = data fit ↑, rough = unphysical banding ↓) ===")
        for r in sorted(results, key=lambda d: -d["corr"]):
            print(f"  {r['method']:14s} corr={r['corr']:.3f}  IoU={r['iou']:.3f}  "
                  f"rough={r['rough']:.3f}")


if __name__ == "__main__":
    main()
