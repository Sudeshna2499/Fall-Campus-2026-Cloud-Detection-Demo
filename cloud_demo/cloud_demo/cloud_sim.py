"""Signal model, features and classifiers for the on-board cloud-screening demo.

A simplified version of what Phi-Sat-1 and Phi-Sat-2 do: look at each image tile on board and
discard the cloudy ones before downlink. Here a tile is 8x8 pixels in 4 bands (R, G, B, NIR),
generated from a surface type plus a cloud fraction. All data are synthetic.

Same code used in the Arduino for simulating the tiles received by the camera
"""
import numpy as np

TW = 8                      # tile is TW x TW pixels
NPIX = TW * TW
GRID_W, GRID_H = 16, 8      # a scene is 16 x 8 tiles
NT = GRID_W * GRID_H
CLASSES = ["clear land", "clear sea", "cloud"]
SURF = ["land", "sea", "sand"]
RAW_TILE_BYTES = NPIX * 4   # 4 bands, 8 bits each
FLAG_BYTES = 1              # per-tile "discarded" marker always sent
DISCARD_P = 0.60            # discard only if P(cloud) is above this (otherwise send)
UNSURE_P = 0.35             # between UNSURE_P and DISCARD_P: sent, flagged as uncertain

VEG = np.array([0.06, 0.11, 0.05, 0.48])
SOIL = np.array([0.22, 0.17, 0.12, 0.30])
SEA = np.array([0.03, 0.05, 0.07, 0.015])
SAND = np.array([0.50, 0.44, 0.34, 0.50])
CLOUD = np.array([0.74, 0.75, 0.78, 0.72])
SIG_SURF = np.array([0.10, 0.03, 0.04])   # pixel texture (relative) for land, sea, sand
SIG_CLOUD = 0.07


def gauss4(rng, shape):
    """Approximate unit Gaussian from four uniforms (same recipe as the Arduino sketch)."""
    u = rng.random(shape + (4,))
    return (u.sum(-1) - 2.0) * np.sqrt(3.0)


def truth_label(surf, alpha):
    return np.where(alpha >= 0.5, 2, np.where(surf == 1, 1, 0))


def make_tiles(surf, v, alpha, rng):
    """surf (n,) 0 land / 1 sea / 2 sand; v (n,) soil fraction for land; alpha (n,) cloud fraction."""
    n = len(surf)
    land = VEG + (SOIL - VEG) * v[:, None]
    base = np.where((surf == 0)[:, None], land, np.where((surf == 1)[:, None], SEA, SAND))
    cb = 0.85 + 0.25 * rng.random(n)
    a = alpha[:, None]
    spec = (1 - a) * base + a * (CLOUD * cb[:, None])
    sig = (1 - a) * SIG_SURF[surf][:, None] + a * SIG_CLOUD
    g = gauss4(rng, (n, NPIX, 1))
    gb = gauss4(rng, (n, NPIX, 4))
    return np.clip(spec[:, None, :] * (1 + sig[:, None, :] * g) * (1 + 0.02 * gb), 0.0, 1.0)


def tile_stats(px):
    """px (n, NPIX, 4) -> raw features m, s, w, ndvi, nir and mean r, g, b."""
    rgb = px[..., :3]
    p = rgb.mean(-1)
    m = p.mean(1)
    s = p.std(1)
    w = ((rgb.max(-1) - rgb.min(-1)) / (p + 0.02)).mean(1)
    r, g, b, nir = px[..., 0].mean(1), px[..., 1].mean(1), px[..., 2].mean(1), px[..., 3].mean(1)
    ndvi = (nir - r) / (nir + r + 0.01)
    return m, s, w, ndvi, nir, r, g, b


def norm_features(m, s, w, ndvi, nir):
    return np.stack([m / 0.5, s / 0.08, w, ndvi, nir / 0.5], -1).astype(np.float32)


def rule_cloud(m, m_thr):
    """Simple baseline: a tile brighter than the threshold is cloud."""
    return m > m_thr


def forward(x, p):
    h = np.maximum(0.0, x @ p["W1"] + p["b1"])
    z = h @ p["W2"] + p["b2"]
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_params(path="model.npz"):
    d = np.load(path)
    return {k: d[k] for k in ("W1", "b1", "W2", "b2")}, float(d["m_thr"])


def _field(rng, nx=GRID_W, ny=GRID_H, k=5):
    ys, xs = np.mgrid[0:ny, 0:nx]
    f = np.zeros((ny, nx))
    for _ in range(k):
        kx, ky = rng.uniform(-0.6, 0.6, 2)
        f += rng.uniform(0.5, 1.0) * np.sin(kx * xs + ky * ys + rng.uniform(0, 6.28))
    return f


def make_scene(seed):
    """Per-tile parameters (surf, v, alpha) for a 16 x 8 scene, flattened row by row."""
    rng = np.random.default_rng(seed)
    sea = _field(rng)
    sand = _field(rng)
    cloud = _field(rng)
    soil = _field(rng)
    surf = np.zeros_like(sea, dtype=int)
    surf[sand > np.quantile(sand, 0.78)] = 2
    surf[sea < np.quantile(sea, 0.30)] = 1
    z = (cloud - np.quantile(cloud, 0.64)) / (0.35 * cloud.std())
    alpha = np.clip(0.5 + z, 0.0, 1.0)
    v = (soil - soil.min()) / (soil.max() - soil.min())
    return surf.ravel(), v.ravel(), alpha.ravel()


def dataset(n, rng):
    surf = rng.choice(3, n, p=[0.5, 0.25, 0.25])
    v = rng.random(n)
    u = rng.random(n)
    alpha = np.where(u < 0.5, 0.0, np.where(u < 0.7, 1.0, rng.random(n)))
    px = make_tiles(surf, v, alpha, rng)
    m, s, w, ndvi, nir, *_ = tile_stats(px)
    return norm_features(m, s, w, ndvi, nir), m, truth_label(surf, alpha)
