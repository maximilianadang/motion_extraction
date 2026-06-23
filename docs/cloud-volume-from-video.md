# Volumetric clouds from a single video: what's actually possible

*Research synthesis — 2026-06-23.*

*Provenance / confidence: **Sections 1–6** (distant sky clouds) come from an adversarially-verified
literature pass (29 sources → 136 candidate claims → 23 confirmed with 2–3 independent votes each);
its automated synthesis step was rate-limited, so this write-up was assembled by hand from the
verified claim set. **Sections 7–8** (stereo & neural depth; small-scale smoke plumes) were added
from a lighter follow-up literature scan — well-sourced but **not** triple-voted. Citations
[10]–[22] belong to that lighter tier.*

*Scope: the headline verdict just below is about **distant sky clouds imaged by one fixed camera**.
It changes substantially for (a) wide-baseline stereo / multi-view geometry (§7) and, especially,
(b) near, optically-thin plumes like **chimney smoke (§8)**, where physics-based volumetric
reconstruction does become feasible.*

## Verdict (the honest bottom line — for distant sky clouds)

**Recovering a physically-accurate 3D density/extinction field from one monocular sky video is
not attainable** — not as an engineering gap, but as an identifiability limit. Every validated
"physically-accurate cloud tomography" method in the literature is built on **simultaneous
multi-angle imaging** plus known sun geometry and calibrated radiometry. The single-view problem
is provably ambiguous. What *is* rigorously obtainable from a single handheld clip is a **tier of
weaker products** (motion/advection fields, relative optical-thickness proxies, approximate
cloud-base height). A *plausible-but-not-measured* 3D volume is obtainable with strong learned
priors — useful for rendering, not for science.

```
                                        physically       single
                                        accurate?        video?
full 3D extinction/LWC field            yes (multi-view)  no
relative column optical-thickness map   proxy only        yes (if exposure locked)
cloud-base height                       yes (stereo/wind) approx (needs wind)
2D motion / advection field             yes               yes   <- doable now
"plausible" 3D volume (prior-driven)    no (not measured) yes
```

## 1. Why single-view physical density is out of reach (theory)

The core obstruction is **provable ambiguity**, not lack of compute. In steady-state passive
imaging there exist *similarity relations* that let you change the scattering parameters at an
interior point **without changing the radiance you measure** — density and phase-function are
fundamentally entangled unless the measurement is *designed* to break the tie [8]. Rigorous
volumetric recovery is normally posed as injecting and measuring light **at the boundary of the
volume**, with unknowns "numbering in the hundreds of thousands" [8]; a single passive sky view is
radiometrically impoverished by comparison.

That is why the field is structured entirely around **angular diversity**:

- Passive single-pixel retrievals assume **1-D plane-parallel-slab** radiative transfer, which
  "fails for vertically-developed 3D clouds" — the explicit motivation for multi-view tomography [1].
- The agreed recovery target is the **3-D volume extinction coefficient σ₃D** (and derived
  microphysics: effective radius, LWC), e.g. "at 40 m resolution from **multi-angle** mono-spectral
  imagery" [2].
- The forward model is **nonlinear**: "image readouts relate nonlinearly to volumetric cloud
  structure by 3-D radiative transfer" via multiple scattering [6].
- Even *with* many views it is **ill-conditioned**: the condition number "increases exponentially
  from well-conditioned κ≈10¹ to very ill-conditioned κ≫10⁵" as clouds get optically thick [2],
  and the Jacobian "becomes increasingly ill-posed as the optical size of the medium increases" [7].

Multiple simultaneous angles are necessary-but-not-sufficient; one angle is far below the bar.

## 2. The three cues — what each buys, where it fails

**Brightness / luminance → optical thickness.** Reflected radiance does encode optical depth, but
the map is **non-injective under multiple scattering** and depends on sun/view geometry. Classical
retrieval (Nakajima–King) needs *two calibrated bands* (a non-absorbing + a liquid-absorbing NIR
band) to separate τ from droplet size. From an **uncalibrated, auto-exposed RGB phone video** you
get at best a **relative** optical-thickness proxy, and only if exposure/ISO/white-balance are
locked and the signal linearized. Brightness alone cannot give absolute density (the
similarity-relation ambiguity [8]).

**Motion / advection as parallax.** The analogy "moving cloud past a fixed camera ≈ fixed cloud,
moving camera" **breaks because clouds deform**. CloudCT is blunt: only over very short windows do
"small warm clouds hardly evolve… at the 20 meter spatial scale" [4] — beyond seconds, internal
evolution violates the rigid multi-view assumption. Advection still yields a useful product: the
**2-D apparent-motion field** via optical flow, and — with an *external* wind estimate — an
approximate **cloud-base height** from angular rate. Rigorous-but-partial, not volumetric.

**Shape / texture.** This is a **prior**, not a measurement — what learned models exploit to
*hallucinate* plausible interior structure (Section 4).

## 3. Classical state of the art (physics-based) — all multi-view

| System | Input requirement | Output | Cite |
|---|---|---|---|
| Levis/Schechner droplet tomography | multi-view **polarimetric** → 3-D polarized RT fit | mass concentration **+** size distribution | [1] |
| **pyshdom / SHDOM** (Evans; Levis) | **multi-angle, multi-spectral** solar-reflected radiance | LWC, effective radius, effective variance | [3] |
| **AT3D** | **multi-angle, multi-pixel** radiances + 3-D RT; ~nine views | σ₃D extinction @ 40 m | [2][7] |
| ECCV'20 multi-view scattering CT | **multi-view, simultaneous** | 3-D extinction + **differentiable monotonicity prior** on extinction vs. altitude | [5] |
| **CloudCT** mission | formation of **ten nanosats**, simultaneous | 3-D reff, LWC, extinction | [4] |

The monotonicity-prior trick [5] is the closest classical analog to "break ambiguity with
physics," but it *supplements* multiple views, it doesn't replace them.

## 4. Learning-based state of the art

The most on-point result is **single-view volumetric reconstruction with a diffusion prior** [9]:
it concedes the problem is "severely under-constrained," then breaks the ambiguity with "an
unconditional diffusion model trained on… 1,000 synthetically simulated volumetric density fields,"
coupled to a **physically-based differentiable volume renderer** that "provides gradients with
respect to light transport" [9] — radiative-transfer-aware, not a geometric NeRF fit. The claim
that it reaches *previously-unachievable single-view quality* came back **verification-inconclusive**
(verifiers abstained under the rate limit) and is shown **on synthetic clouds** — treat as
"promising, not validated on real phone video."

Adjacent tooling: **Mitsuba 3** (differentiable path tracer for inverse volume rendering),
**PIVOT-CT** (DNN cloud CT — still needs "10 multiview images + camera poses + sun direction" [6]),
and fluid/smoke tomography (**ScalarFlow**). All reinforce that few/single-view volume recovery
leans on priors or many views.

## 5. Where "motion extraction" fits

The invert + time-delay + 50% overlay is a **temporal-gradient / edge visualizer** — it highlights
*where and how much changed between frames*. It is **not a depth or density cue** by itself. Its
value here is as a **front-end for motion estimation** (isolating moving cloud boundaries that
optical flow then quantifies) and for segmentation/QA. Don't expect it to contribute volumetric
information.

## 6. A pragmatic, honestly-scoped pipeline

**Tier 1 — rigorous-but-partial (defensible):**
1. **Stabilize** + **sky/cloud segmentation** (dataset: SWIMSEG).
2. **Optical flow** (RAFT, or OpenCV Farnebäck) → 2-D advection field. A real apparent-motion
   measurement; the motion-extraction step feeds it.
3. **Relative optical-thickness proxy**: lock exposure/ISO/WB on capture, linearize RGB, normalize.
   Label it *relative*; not calibrated density.
4. **Approximate cloud-base height**: combine tracked angular velocity with an external wind speed
   (nearest sounding/METAR). Flag as approximate (single camera, no true stereo baseline).

**Tier 2 — plausible-only 3D (label as such):**
5. Feed the single view + τ-proxy into a **diffusion-prior + differentiable volume renderer** (the
   [9] recipe; renderer in **Mitsuba 3**). Output: a 3-D density *consistent with your one view* —
   for rendering/VFX, **not** a measurement.

**If capture can change** (biggest leverage): add a **second synchronized camera** with a known
baseline (or a deliberate translating sweep over seconds while the cloud is quasi-static [4]) →
the regime where **pyshdom/AT3D** can attempt genuine σ₃D retrieval.

**Tools:** `pyshdom` / `AT3D`, SHDOM, **Mitsuba 3**, ScalarFlow (reference volumes), SWIMSEG
(segmentation), OpenCV/RAFT (flow).

## 7. Stereo imaging & neural monocular depth (Depth Anything family)

Both of these recover **surface geometry**, never interior density — keep that distinction front
of mind. They live in the "geometry" lane, not the "physical density" lane.

### 7a. Stereo photogrammetry of clouds — real, validated, but baseline-bound

Ground-based cloud **stereophotogrammetry** is a mature technique: triangulate matched pixels
between two rectified, widely-separated cameras to get **cloud-base height + cloud velocity
(wind) fields** [10][11][12]. Reported accuracy is good — an 800 m baseline of consumer cameras
gave cloud-base height over a ~100° field of view "with errors well below 5%" [12]. A nice
calibration trick: photograph the **night starry sky** to recover each camera's unknown optical-axis
direction (absolute orientation) [13].

The hard constraint is the **baseline**. Stereo range error grows as

```
δZ ≈ Z² · δd / (f · B)        Z = cloud range, f = focal (px), B = baseline, δd = match error (px)
disparity d = f · B / Z
```

Clouds sit at Z ≈ 1–10 km, so the geometry is brutal for a single handheld camera. Illustrative
numbers (iPhone-ish f ≈ 1400 px, cumulus base Z = 2000 m):

| Baseline B | Disparity at 2 km | Verdict |
|---|---|---|
| 0.3 m (handheld "small-motion stereo") | **0.2 px** | unmeasurable |
| 14 m | 10 px | ~100 m depth error; marginal |
| 100 m – 800 m (real all-sky stereo nets) [11][12] | tens–hundreds px | usable; the actual regime |
| 5.6 km (historic film baseline) [12] | — | high-altitude clouds |

So: **handheld single-camera "stereo from small motion" cannot triangulate clouds** — the baseline
is hopelessly small versus cloud distance. Two real cameras separated by ≥ hundreds of meters
work. And even then you recover the **cloud-base surface + wind, not the 3-D density**, because
clouds are semi-transparent and **view-dependent** (a cloud point's appearance changes between the
two angles due to scattering), which violates the brightness-constancy assumption dense stereo
matching relies on — so matching is fragile and works best on textured cumulus edges [10].

### 7b. Neural monocular depth — Depth Anything V1/V2/**3**, Video-Depth-Anything

Important: **Depth Anything 3** (ByteDance, released 2025-11-14) is *not* the same animal as V1/V2.
It is a **multi-view visual-geometry transformer** (DUSt3R/VGGT lineage) that accepts "arbitrary
numbers of visual inputs with or without known camera poses," from "single view to multiple views"
and video, and outputs spatially-consistent geometry, depth, **camera pose**, and even 3D Gaussian
Splatting parameters [14]. V1 (CVPR'24) / V2 (NeurIPS'24) are the *monocular relative-depth* models
(DINOv2 encoder, metric variants available) [15]; **Video-Depth-Anything** (CVPR'25) adds temporal
consistency for long videos.

Why none of them deliver your goal, for three structural reasons:

1. **Single-surface assumption.** They predict *one* depth per pixel = the opaque surface that pixel
   images. A cloud pixel integrates radiance along a path through a scattering volume, so "the
   depth" is undefined. Even a perfect output is the **wrong type of quantity** — a surface, not a
   density field.
2. **Out-of-distribution.** Sky/clouds are badly underrepresented in training (DA3's benchmarks are
   Colosseum, vehicles, dance, animals — all opaque structure [14]). Documented behavior: Depth
   Anything gives "blurry estimates" for sky and diffusion models like Marigold "struggle with sky
   regions," pushing them to infinity [16]. An up-at-the-sky shot with no terrestrial anchor is
   close to worst-case.
3. **Scale/shift ambiguity.** Relative models give unknown scale+shift; metric variants
   (`metric = focal·output/300` for DA3 [14]) bake in opaque-surface priors that don't transfer to
   sky. No reliable monocular prior pins cloud distance.

What DA3's **multi-view mode** *could* buy you, honestly bounded: feed it your video frames as
"multiple views." But a **fixed** camera has ≈zero translation baseline (no parallax), and the cloud
**deforms** between frames — so the static-rigid-scene assumption underlying multi-view geometry is
violated twice. If instead you **walk/translate** the camera a few meters over a few seconds while
the cloud is quasi-static [4], DA3/VGGT-style pose+geometry estimation becomes meaningful — but it
still recovers cloud **surface** points (relative, OOD-fragile), not volumetric density.

### 7c. Net

- **Want geometry (base height, surfaces, wind)?** Rigorous path = **two wide-baseline cameras
  (≥ hundreds of m), zenith-pointing, star-calibrated → dense stereo** [10][12]. Validated to <5%
  on cloud-base height. DA3 in multi-view mode is the modern learning-based analog and a fast
  experiment on a *translating* capture — but treat its output as relative/plausible surface
  geometry, not metric, not volume.
- **Want physically-accurate volumetric density?** Neither stereo nor monocular neural depth gets
  there. That still requires the multi-angle radiometric **tomography** of Sections 1–4
  (pyshdom/AT3D). Neural depth's legitimate role is as a **prior/initializer** that regularizes a
  physics-based or diffusion-prior volume model (Section 4) — the "shape/texture cue," operationalized.

## 8. Small-scale plumes: smoke from a chimney (the verdict flips)

Everything above is pessimistic because *sky clouds* are far away and optically thick. A **chimney
plume changes both variables**, and the feasibility flips from "no" to "largely yes." *(Note: the
sources in this section come from a focused literature scan, not the same adversarial-verification
pass as Sections 1–7 — treat as well-sourced but not triple-voted.)*

**Why smoke is fundamentally easier — two independent reasons:**

1. **Distance → stereo/parallax actually works.** A plume is at ~10–200 m, not 1–10 km. Re-running
   the disparity math (`d = f·B/Z`, f ≈ 1400 px): a 0.3 m handheld baseline gives **~14 px disparity
   at Z = 30 m** (vs 0.2 px at a 2 km cloud) — readily measurable. So handheld *small-motion stereo*,
   a couple of phones, or a short translating sweep all yield real parallax on a plume.
2. **Optical thinness → the tomography is well-posed.** Thin smoke (and flame) is dominated by
   **single scattering / emission–absorption**, so the forward model is ~linear (Beer–Lambert line
   integrals), i.e. classic CT — *well-posed* with enough angles. That's the opposite of the
   multiply-scattering, optically-thick cloud regime whose Jacobian condition number blows up to
   κ≫10⁵ [2]. Flame **chemiluminescence tomography** routinely reconstructs 3-D fields from just
   **3–36 views (statistics from as few as 6)** for exactly this reason [22].

**Methods that already recover physically-meaningful 3-D density (often + velocity) from video:**

| Method | Input | Output | Cite |
|---|---|---|---|
| **ScalarFlow** (Eckert/Um/Thuerey '19) | **5 commodity cameras** on a 120° arc | time-resolved 3-D **density + velocity** via physics-/simulation-constrained tomography; dataset + capture code public | [17] |
| **Global Transport** (Franz et al. CVPR'21) | **as few as a single view** | smoke density + velocity, via global-transport physics + learned self-supervision + differentiable rendering | [18] |
| **Physics-Informed Neural Fields** (Chu et al. SIGGRAPH'22) | **sparse** video frames | density + velocity, Navier–Stokes-constrained; disentangles density–color ambiguity | [19] |
| **SmokeSVD** ('25) | **single video** | dynamic smoke via diffusion-based novel-view synthesis + physics-guided refinement | [20] |
| **WildSmoke** ('25) | **single video "in the wild"** | ready-to-use dynamic 3-D smoke asset | [21] |
| **Flame chemiluminescence tomography** | 3–36 calibrated views | 3-D intensity/density field of a participating medium | [22] |

So the controllable, multi-camera version (ScalarFlow-style, ~5 cheap cameras) gives a genuinely
**physics-based volumetric reconstruction** — the thing that's impossible for sky clouds — and even
**single-video** methods produce convincing dynamic volumes, because the unseen sides are filled by
strong physics/generative priors rather than left ambiguous.

**Honest caveats:**
- **Absolute units.** These recover a self-consistent *scalar density* field (graphics-grade),
  not necessarily calibrated kg/m³ mass concentration without extra radiometric calibration.
- **Single-view is still prior-driven.** Global Transport / SmokeSVD / WildSmoke regularize the
  ambiguity with transport physics or diffusion priors — multi-view (2–5 cams) is far more reliable.
- **Smoke type matters.** White steam/condensation = scattering; sooty exhaust = absorbing; the
  forward model differs, and an optically *thick*, dark plume near the stack regresses toward the
  hard (cloud-like) regime.
- **Turbulence + non-rigidity.** Plumes evolve fast; a single moving camera sees a *deforming*
  volume (same issue as clouds), so multi-view wants **synchronized** cameras — or use the
  transport-physics methods that solve density **and** velocity jointly over time.

**Practical recipe for a chimney plume:**
1. Easiest rigorous route: **2–5 synchronized phones** on a short arc (even a 0.5–2 m total span),
   plume **backlit against bright sky** for a clean matte → segment → ScalarFlow-style
   differentiable-rendering tomography ([17] code: `tum-pbs/reconstructScalarFlows`).
2. Single-clip route: one steady video → **Global Transport / WildSmoke**-style reconstruction
   [18][21] for a plausible dynamic volume (great for VFX; physics-consistent, not calibrated).
3. Your existing **motion-extraction + optical flow** is a useful front-end here too — it isolates
   the moving plume from the static background and seeds the velocity field.

## References (verified sources)

- [1] Levis et al., *3D tomography of cloud droplets* — https://arxiv.org/pdf/2005.11423
- [2] *AT3D ill-posedness & σ₃D retrieval*, AMT 16/3931/2023 — https://amt.copernicus.org/articles/16/3931/2023/
- [3] **pyshdom** — https://github.com/aviadlevis/pyshdom
- [4] Tzabari et al., *CloudCT* — https://omershubi.github.io/publication/tzabari-2021-cloudct/
- [5] *Multi-view scattering CT w/ monotonicity prior*, ECCV 2020 — https://link.springer.com/chapter/10.1007/978-3-030-58523-5_17
- [6] *PIVOT-CT* — https://arxiv.org/pdf/2411.04682
- [7] *AT3D open framework*, AMT 16/1803/2023 — https://amt.copernicus.org/articles/16/1803/2023/
- [8] Gkioulekas et al., *Heterogeneous Inverse Scattering* — https://www.researchgate.net/publication/308277200
- [9] *Single-view volumetric reconstruction with diffusion prior + differentiable rendering* — https://arxiv.org/pdf/2501.05226
- [10] Beekmans et al., *Cloud Photogrammetry with Dense Stereo for Fisheye Cameras*, ACP 2016 — https://acp.copernicus.org/preprints/acp-2016-319/acp-2016-319-manuscript-version2.pdf
- [11] *Stereo cloud base height estimation using a pair of all-sky cameras* — https://www.researchgate.net/publication/374529682
- [12] *Advances in cloud base height and wind speed measurement through stereophotogrammetry with low-cost consumer cameras* (800 m baseline, <5%) — https://www.researchgate.net/publication/265736250
- [13] *Stereoscopic ground-based determination of cloud base height* (night-sky star calibration) — https://www.researchgate.net/publication/311169022
- [14] **Depth Anything 3** (ByteDance, 2025-11-14) — https://depth-anything-3.github.io/ · https://github.com/ByteDance-Seed/Depth-Anything-3
- [15] **Depth Anything V2** (NeurIPS 2024) — https://github.com/DepthAnything/Depth-Anything-V2 · *Video-Depth-Anything* (CVPR 2025) — https://github.com/DepthAnything/Video-Depth-Anything
- [16] *Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions* (sky-region failure) — https://arxiv.org/html/2407.16698v1
- [17] Eckert, Um, Thuerey, *ScalarFlow* (SIGGRAPH Asia 2019) — https://arxiv.org/abs/2011.10284 · code: https://github.com/tum-pbs/reconstructScalarFlows
- [18] Franz, Solenthaler, Thuerey, *Global Transport for Fluid Reconstruction with Learned Self-Supervision* (CVPR 2021) — https://www.researchgate.net/publication/355882687
- [19] Chu et al., *Physics-Informed Neural Fields for Smoke Reconstruction with Sparse Data* (SIGGRAPH 2022) — https://arxiv.org/abs/2206.06577
- [20] *SmokeSVD: Smoke Reconstruction from a Single View …* (2025) — https://arxiv.org/abs/2507.12156
- [21] *WildSmoke: Ready-to-Use Dynamic 3D Smoke Assets from a Single Video in the Wild* (2025) — https://arxiv.org/html/2509.11114v1
- [22] *A Survey for 3D Flame Chemiluminescence Tomography: Theory, Algorithms, and Applications* — https://www.frontiersin.org/journals/photonics/articles/10.3389/fphot.2022.845971/full

*Also surfaced: Yoav Schechner's scattering-tomography page (webee.technion.ac.il), Mitsuba 3
(github.com/mitsuba-renderer/mitsuba3), ScalarFlow, SWIMSEG (malea.winkler.site/swimseg.html), and
several AMT / Remote Sensing cloud-base/stereo papers.*
