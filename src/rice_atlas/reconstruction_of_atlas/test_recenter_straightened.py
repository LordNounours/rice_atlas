"""
Recenter a multi-slice TIFF using FFT phase-correlation on every k-th slice,
then linearly interpolate shifts for intermediate slices.

Assumptions (your spec):
- The first 1% of NON-NULL slices are already well centered.
- The last NON-NULL slices are already well centered.
- Max drift is small (a few pixels), so we restrict estimated shifts to +/- max_shift.
- You will estimate every k slices (default k=5) and interpolate in between.

Deps: numpy, tifffile, scipy
  pip install numpy tifffile scipy
"""

from __future__ import annotations
import numpy as np
import tifffile as tiff
from scipy.ndimage import shift as ndi_shift


def _gaussian_weight(shape, sigma=10.0):
    """
    Create a 2D Gaussian weight centered on the image, with given sigma.
    Higher weight at center, decreasing towards edges.
    """
    h, w = shape
    y, x = np.ogrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    
    r2 = (y - cy)**2 + (x - cx)**2
    gauss = np.exp(-r2 / (2.0 * sigma**2))
    
    return gauss.astype(np.float32)


def _hann2(shape):
    hy = np.hanning(shape[0]).astype(np.float32)
    hx = np.hanning(shape[1]).astype(np.float32)
    return hy[:, None] * hx[None, :]


def _highpass(I: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    # Keep it minimal: mean removal + optional gentle highpass via FFT notch
    # (for small 128x128, mean removal is often enough)
    J = I.astype(np.float32)
    J -= J.mean()
    return J


def phase_corr_shift(I: np.ndarray, J: np.ndarray, eps: float = 1e-7, window: bool = True, center_weight_sigma: float = 10.0):
    """
    Return (dx, dy), score from phase correlation (integer shift).
    dx>0 means J is shifted right relative to I (so to align J onto I you shift J by (-dx,-dy)).
    
    Args:
        I, J: 2D arrays to correlate
        eps: Small value for numerical stability
        window: Apply Hann window
        center_weight_sigma: Sigma for Gaussian weighting (0 to disable center weighting)
    """
    I = _highpass(I)
    J = _highpass(J)

    # Apply center weighting (Gaussian mask)
    if center_weight_sigma > 0:
        G = _gaussian_weight(I.shape, sigma=center_weight_sigma)
        I = I * G
        J = J * G

    # Apply window (Hann)
    if window:
        W = _hann2(I.shape)
        I = I * W
        J = J * W

    F = np.fft.fft2(I)
    G_fft = np.fft.fft2(J)
    R = F * np.conj(G_fft)
    R /= (np.abs(R) + eps)
    r = np.abs(np.fft.ifft2(R))

    py, px = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    dy = py if py < H // 2 else py - H
    dx = px if px < W // 2 else px - W

    score = r[py, px] / (r.mean() + eps)
    return (dx, dy), float(score)


def _clip_shift(dxdy, max_shift):
    dx, dy = dxdy
    dx = int(np.clip(dx, -max_shift, max_shift))
    dy = int(np.clip(dy, -max_shift, max_shift))
    return dx, dy


def recenter_tiff_phasecorr(
    in_path: str,
    out_path: str,
    step: int = 5,
    max_shift: int = 5,
    anchor_frac: float = 0.01,
    null_thresh: int = 0,
    interpolation_order: int = 1,  # 0 nearest, 1 bilinear (good), 3 cubic
    center_weight_sigma: float = 10.0,  # NEW: Gaussian sigma for center weighting
):
    """
    - Reads multi-slice TIFF (Z, Y, X)
    - Finds non-null z-range
    - Uses first anchor_frac of non-null slices as "start anchor" (no correction)
      and last anchor_frac as "end anchor" (no correction)
    - Estimates cumulative drift on sampled slices (every 'step') by phase-corr slice-to-slice
    - Enforces boundary: correction shift = 0 at start anchor and at end anchor (linear drift removal)
    - Interpolates shifts for all z between anchors
    - Applies shifts to recenter (negative of estimated drift)
    - Writes corrected TIFF
    
    Args:
        center_weight_sigma: Sigma for Gaussian weighting of center pixels (0 to disable)
    """
    vol = tiff.imread(in_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected (Z,Y,X) TIFF. Got shape={vol.shape}")

    Z, Y, X = vol.shape

    # 1) Find non-null slices (your "non nulles")
    nonnull = np.array([np.any(vol[z] > null_thresh) for z in range(Z)], dtype=bool)
    if not np.any(nonnull):
        raise ValueError("No non-null slices found.")

    z_non = np.where(nonnull)[0]
    z0, z1 = int(z_non[0]), int(z_non[-1])  # inclusive bounds of non-null region
    L = z1 - z0 + 1

    # 2) Define anchors
    a = max(1, int(np.ceil(anchor_frac * L)))
    start_anchor_end = z0 + a - 1
    end_anchor_start = z1 - a + 1
    if end_anchor_start <= start_anchor_end + 1:
        # too short, shrink anchors
        mid = (z0 + z1) // 2
        start_anchor_end = min(mid, start_anchor_end)
        end_anchor_start = max(mid + 1, end_anchor_start)

    # Region to correct (strictly between anchors)
    corr_start = start_anchor_end + 1
    corr_end = end_anchor_start - 1

    # If nothing to correct, just copy
    if corr_end < corr_start:
        tiff.imwrite(out_path, vol)
        return {
            "in_path": in_path,
            "out_path": out_path,
            "z0": z0,
            "z1": z1,
            "start_anchor": (z0, start_anchor_end),
            "end_anchor": (end_anchor_start, z1),
            "note": "No middle region to correct; copied as-is.",
        }

    # 3) Sample indices for estimation, always include corr_start and corr_end
    sample = list(range(corr_start, corr_end + 1, step))
    if sample[0] != corr_start:
        sample = [corr_start] + sample
    if sample[-1] != corr_end:
        sample.append(corr_end)
    sample = np.array(sample, dtype=int)

    # 4) Estimate cumulative drift on sampled slices (relative to corr_start)
    # drift[z] = how much the content of slice z is shifted relative to slice corr_start (integer pixels)
    drift = np.zeros((len(sample), 2), dtype=np.float32)
    scores = np.zeros(len(sample), dtype=np.float32)

    # Start: drift=0 at corr_start by definition
    prev_idx = sample[0]
    prev_img = vol[prev_idx]
    scores[0] = 1.0

    for i in range(1, len(sample)):
        cur_idx = sample[i]
        cur_img = vol[cur_idx]

        (dx, dy), sc = phase_corr_shift(prev_img, cur_img, window=True, center_weight_sigma=center_weight_sigma)
        dx, dy = _clip_shift((dx, dy), max_shift=max_shift)

        drift[i] = drift[i - 1] + np.array([dx, dy], dtype=np.float32)
        scores[i] = sc

        prev_idx = cur_idx
        prev_img = cur_img

    # 5) Enforce boundary condition: correction shift is 0 at BOTH ends of corrected region
    # Our "drift" is measured relative to corr_start, so drift[0]=0 already.
    # We want drift at corr_end to be also 0 (because end anchor is good, so no net correction at the end).
    # Remove linear component so drift_adj[0]=0 and drift_adj[-1]=0.
    end_drift = drift[-1].copy()
    t = np.linspace(0.0, 1.0, len(sample), dtype=np.float32)[:, None]
    drift_adj = drift - t * end_drift  # now endpoints are (0,0)

    # 6) Interpolate drift for every z between corr_start..corr_end (inclusive)
    # We do x(z), y(z) separately with np.interp (piecewise linear)
    z_all = np.arange(corr_start, corr_end + 1, dtype=np.int32)
    dx_all = np.interp(z_all, sample, drift_adj[:, 0]).astype(np.float32)
    dy_all = np.interp(z_all, sample, drift_adj[:, 1]).astype(np.float32)

    # 7) Build output volume
    out = vol.copy()

    # Apply shifts: to recenter slice z, shift it by (-dx, -dy)
    for zz, dxv, dyv in zip(z_all, dx_all, dy_all):
        out[zz] = ndi_shift(
            out[zz].astype(np.float32),
            shift=(-dyv, -dxv),  # scipy uses (y,x)
            order=interpolation_order,
            mode="nearest",
            prefilter=False if interpolation_order <= 1 else True,
        ).astype(vol.dtype)

    # Anchors untouched by design: [z0..start_anchor_end] and [end_anchor_start..z1]
    tiff.imwrite(out_path, out)

    return {
        "in_path": in_path,
        "out_path": out_path,
        "shape": (int(Z), int(Y), int(X)),
        "non_null_range": (int(z0), int(z1)),
        "start_anchor": (int(z0), int(start_anchor_end)),
        "end_anchor": (int(end_anchor_start), int(z1)),
        "corrected_range": (int(corr_start), int(corr_end)),
        "step": int(step),
        "max_shift": int(max_shift),
        "center_weight_sigma": float(center_weight_sigma),
        "scores_summary": {
            "min": float(np.min(scores)),
            "median": float(np.median(scores)),
            "max": float(np.max(scores)),
        },
        "end_drift_before_boundary_fix": (float(end_drift[0]), float(end_drift[1])),
    }


if __name__ == "__main__":
    info = recenter_tiff_phasecorr(
        in_path="/media/rfernandez/Crucial X9/Test_Charlotte_2026_01_root_and_leave/output/T0-0_Fa/T0-0_Fa_straightened.tif",
        out_path="/media/rfernandez/Crucial X9/Test_Charlotte_2026_01_root_and_leave/output/T0-0_Fa/T0-0_Fa_straightened_recentered.tif",
        step=5,
        max_shift=5,
        anchor_frac=0.01,
        null_thresh=0,
        interpolation_order=1,
        center_weight_sigma=5.0,  # NEW: Gaussian weighting with sigma=10 pixels
    )
    print(info)
