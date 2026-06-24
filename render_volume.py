"""Visualize the extracted volumetric plume prior.

Loads out/volumetric/{volume_density.npy, volume_meta.json, plume_profile.csv}
and renders a multi-panel figure:

  1. 3D point cloud of the reconstructed plume, placed along the REAL bent
     centerline in world coordinates (downwind x, crosswind, height).
  2-4. Orthogonal maximum-intensity projections (side / top / end-on).
  5. Re-projection: integrate the volume through depth -> synthetic camera
     silhouette (validates the volume reproduces what the camera saw).
  6. Centerline + sigma(s) profile.

Usage:  uv run render_volume.py [out/volumetric]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def load(d: Path):
    vol = np.load(d / "volume_density.npy")
    meta = json.loads((d / "volume_meta.json").read_text())
    rows = list(csv.DictReader((d / "plume_profile.csv").open()))
    x = np.array([float(r["centerline_downwind_x_m"]) for r in rows])
    z = np.array([float(r["centerline_height_m"]) for r in rows])
    sig = np.array([float(r["sigma_m"]) for r in rows])
    return vol, meta, x, z, sig


def main(argv):
    d = Path(argv[0]) if argv else Path("out/volumetric")
    vol, meta, cx, cz, sig = load(d)
    n_s, n_a, n_b = vol.shape
    R = float(meta["cross_section_radius_m"])
    a = np.linspace(-R, R, n_a)        # crosswind / depth
    b = np.linspace(-R, R, n_b)        # in-plane transverse
    # align profile length with grid stations
    m = min(n_s, len(cx))
    vol, cx, cz, sig = vol[:m], cx[:m], cz[:m], sig[:m]

    fig = plt.figure(figsize=(15, 9))

    # 1) 3D world-placed point cloud
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ii, ja, jb = np.where(vol > 0.08)
    dens = vol[ii, ja, jb]
    if len(ii) > 35000:
        sub = np.random.default_rng(0).choice(len(ii), 35000, replace=False)
        ii, ja, jb, dens = ii[sub], ja[sub], jb[sub], dens[sub]
    X, Y, Z = cx[ii], a[ja], cz[ii] + b[jb]
    ax.scatter(X, Y, Z, c=dens, s=3, alpha=0.12, cmap="inferno")
    ax.set_xlabel("downwind x [m]"); ax.set_ylabel("crosswind [m]"); ax.set_zlabel("height [m]")
    ax.set_title("3D reconstructed plume (world-placed)")
    px, py, pz = np.ptp(X), np.ptp(Y), np.ptp(Z)
    rng = max(px, py, pz, 1e-3)
    ax.set_box_aspect((max(px, 1e-3) / rng, max(py, 1e-3) / rng, max(pz, 1e-3) / rng))
    ax.view_init(elev=18, azim=-70)

    # 2) side MIP (look along crosswind/depth) -> downwind x vs transverse
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(vol.max(axis=1).T, origin="lower", cmap="inferno", aspect="auto")
    ax.set_title("Side max-projection (along depth)")
    ax.set_xlabel("station s"); ax.set_ylabel("transverse b")

    # 3) top MIP (look along transverse) -> downwind x vs crosswind
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(vol.max(axis=2).T, origin="lower", cmap="inferno", aspect="auto")
    ax.set_title("Top max-projection (along vertical)")
    ax.set_xlabel("station s"); ax.set_ylabel("crosswind a")

    # 4) end-on cross-section MIP (look along the plume)
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(vol.max(axis=0), origin="lower", cmap="inferno",
              extent=[-R, R, -R, R])
    ax.set_title("End-on max cross-section")
    ax.set_xlabel("crosswind a [m]"); ax.set_ylabel("transverse b [m]")

    # 5) re-projection: integrate through depth -> synthetic silhouette
    ax = fig.add_subplot(2, 3, 5)
    reproj = vol.sum(axis=1).T          # (n_b, n_s) column density seen by camera
    ax.imshow(reproj, origin="lower", cmap="magma", aspect="auto")
    ax.set_title("Re-projected silhouette (∫ through depth)")
    ax.set_xlabel("station s"); ax.set_ylabel("transverse b")

    # 6) centerline + sigma envelope in world coords
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(cx, cz, "k-", lw=1.5, label="centerline")
    ax.fill_between(cx, cz - 2 * sig, cz + 2 * sig, alpha=0.25, color="tab:orange",
                    label="±2σ")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("downwind x [m]"); ax.set_ylabel("height [m]")
    ax.set_title("Centerline + spread (world)"); ax.legend(fontsize=8)

    fig.suptitle(f"Volumetric plume prior — {meta.get('source_video', '')}  "
                 f"(grid {n_s}×{n_a}×{n_b}, relative density)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = d / "render_volume.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")

    # dedicated, larger 3D view from three angles
    box = (max(px, 1e-3) / rng, max(py, 1e-3) / rng, max(pz, 1e-3) / rng)
    fig2 = plt.figure(figsize=(16, 5.5))
    for k, (el, az, name) in enumerate([(18, -72, "perspective"),
                                        (2, -90, "side (camera-like)"),
                                        (88, -90, "top-down")]):
        ax = fig2.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(X, Y, Z, c=dens, s=4, alpha=0.14, cmap="inferno")
        ax.set_box_aspect(box)
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("downwind x [m]"); ax.set_ylabel("crosswind [m]")
        ax.set_zlabel("height [m]")
        ax.set_title(name)
    fig2.suptitle("3D reconstructed plume volume — multiple viewpoints", fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    out3d = d / "render_volume_3d.png"
    fig2.savefig(out3d, dpi=130)
    plt.close(fig2)
    print(f"wrote {out3d}")
    print(f"voxels > 0.08: {int((vol > 0.08).sum())}; "
          f"world extent x[{cx.min():.2f},{cx.max():.2f}] "
          f"z[{cz.min():.2f},{cz.max():.2f}] m")


if __name__ == "__main__":
    main(sys.argv[1:])
