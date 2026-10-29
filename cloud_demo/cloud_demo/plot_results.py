"""Results figure for the on-board cloud-screening demo.

  python3 plot_results.py --sim --scenes 100 --out results.png     # simulated, no hardware
  python3 plot_results.py --csv run.csv --out results.png          # a run logged from the board:
                                                                   #   python3 live_plot.py --port COM3 --log run.csv
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from cloud_sim import GRID_W, GRID_H, NT, DISCARD_P, UNSURE_P, load_params
from live_plot import sim_source, status, COLORS, LOG_COLS

INK, MUTED = "#1B2433", "#55607A"
C_GOOD, C_DROP, C_LOST, C_WASTE = "#4C9F70", "#8A8F9E", "#D1342F", "#F2A33A"


def load_csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {k: d[k] for k in d.dtype.names}


def load_sim(n_scenes):
    params, m_thr = load_params()
    src = sim_source(params, m_thr)
    rows = [next(src) for _ in range(n_scenes * NT)]
    out = {k: np.array([r[k] for r in rows]) for k in ("idx", "surf", "truth", "m", "s", "w", "rule", "pc", "send", "raw", "tx")}
    out["cloud_pct"] = np.array([r["cpct"] for r in rows])
    out["r"], out["g"], out["b"] = (np.array([r["rgb"][i] for r in rows]) for i in range(3))
    out["pcloud"] = out.pop("pc")
    out["raw_bytes"] = out.pop("raw"); out["tx_bytes"] = out.pop("tx")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--sim", action="store_true")
    ap.add_argument("--scenes", type=int, default=100); ap.add_argument("--out", default="results.png")
    a = ap.parse_args()
    d = load_csv(a.csv) if a.csv else load_sim(a.scenes)
    n = len(d["idx"]); n_sc = n // NT
    if n_sc < 1:
        raise SystemExit(f"need at least one full scene ({NT} tiles), got {n}")
    n = n_sc * NT
    d = {k: v[:n] for k, v in d.items()}
    truth = d["truth"].astype(int); cloud = truth == 2
    dis_rule = d["rule"] == 2; dis_mlp = d["send"] == 0
    p = d["pcloud"]

    fig = plt.figure(figsize=(15, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28, left=0.10, right=0.98, top=0.86, bottom=0.08)
    fig.text(0.10, 0.95, "On-board cloud screening: what reaches the ground", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.10, 0.915, f"Synthetic image tiles, {n_sc} scenes x {NT} tiles. A tiny neural network (75 parameters) versus a simple brightness rule.",
             fontsize=11, color=MUTED)

    # last full scene
    sl = slice(n - NT, n)
    rgb = np.clip(np.stack([d["r"][sl], d["g"][sl], d["b"][sl]], -1) * 1.15, 0, 1) ** 0.6
    cmap = ListedColormap(COLORS)
    st_rule = np.array([status(t, x) for t, x in zip(truth[sl], dis_rule[sl])])
    st_mlp = np.array([status(t, x) for t, x in zip(truth[sl], dis_mlp[sl])])
    unsure = (~dis_mlp[sl]) & (p[sl] > UNSURE_P)
    for k, (title, img, kind) in enumerate([("The scene the satellite sees", rgb.reshape(GRID_H, GRID_W, 3), "rgb"),
                                           ("Brightness rule decides", st_rule.reshape(GRID_H, GRID_W), "st"),
                                           ("Neural network decides", st_mlp.reshape(GRID_H, GRID_W), "st")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, interpolation="nearest", **({} if kind == "rgb" else dict(cmap=cmap, vmin=0, vmax=4)))
        ax.set_xticks(np.arange(-.5, GRID_W, 1), minor=True); ax.set_yticks(np.arange(-.5, GRID_H, 1), minor=True)
        ax.grid(which="minor", color="#8892A6" if kind == "rgb" else "white", lw=1.2); ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, fontsize=12, color=INK, loc="left", fontweight="bold")
        if k == 2:
            ys, xs = np.where(unsure.reshape(GRID_H, GRID_W))
            ax.scatter(xs, ys, s=70, facecolors="none", edgecolors=INK, linewidths=1.3)
            ax.legend(handles=[Patch(color=C_GOOD, label="clear, sent"), Patch(color=C_DROP, label="cloud, discarded"),
                               Patch(color=C_LOST, label="clear, discarded (lost)"), Patch(color=C_WASTE, label="cloud, sent"),
                               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="sent, unsure")],
                      fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False)

    # outcomes per method
    ax = fig.add_subplot(gs[1, 0])
    names = ["Brightness\nrule", "Neural\nnetwork"]
    parts = []
    for dis in (dis_rule, dis_mlp):
        parts.append([(dis & cloud).sum(), (~dis & ~cloud).sum(), (dis & ~cloud).sum(), (~dis & cloud).sum()])
    parts = np.array(parts) / n * 100
    left = np.zeros(2)
    for j, (lab, col) in enumerate([("cloud discarded", C_DROP), ("clear sent", C_GOOD), ("clear lost", C_LOST), ("cloud sent", C_WASTE)]):
        ax.barh(names, parts[:, j], left=left, color=col, label=lab, height=0.55)
        for i in range(2):
            if parts[i, j] > 4:
                ax.text(left[i] + parts[i, j] / 2, i, f"{parts[i, j]:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left += parts[:, j]
    lost = [(dis & ~cloud).sum() / max((~cloud).sum(), 1) * 100 for dis in (dis_rule, dis_mlp)]
    for i, v in enumerate(lost):
        ax.text(101, i, f"{v:.1f}% of clear\ntiles lost", va="center", fontsize=9, color=C_LOST)
    ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.invert_yaxis()
    ax.set_xlabel("share of all tiles (%)"); ax.set_title("Every tile, by outcome", fontsize=12, loc="left", color=INK, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.2), frameon=False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # confidence histogram
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 26)
    ax.hist(p[~cloud], bins=bins, color=C_GOOD, alpha=0.85, label="clear tiles")
    ax.hist(p[cloud], bins=bins, color=C_DROP, alpha=0.85, label="cloudy tiles")
    ax.set_yscale("log"); ax.set_ylim(0.7, 2e5)
    ax.axvspan(UNSURE_P, DISCARD_P, color=C_WASTE, alpha=0.18, lw=0)
    ax.axvline(DISCARD_P, color=INK, ls="--", lw=1)
    ax.text(DISCARD_P + 0.015, 1.5e3, "discard\nabove", fontsize=8, color=INK, va="top")
    ax.text((UNSURE_P + DISCARD_P) / 2, 1.2e3, "unsure,\nsent", fontsize=8, color="#9A6A10", ha="center")
    ax.set_xlabel("model's P(cloud)"); ax.set_ylabel("tiles (log scale)")
    ax.set_title("How sure is the model?", fontsize=12, loc="left", color=INK, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    # bytes per scene
    ax = fig.add_subplot(gs[1, 2])
    raw = d["raw_bytes"].reshape(n_sc, NT)[:, -1]; tx = d["tx_bytes"].reshape(n_sc, NT)[:, -1]
    tot_raw, tot_tx = raw.sum(), tx.sum()
    ax.bar(["send every tile", "send what the\nmodel keeps"], [raw.mean() / 1024, tx.mean() / 1024], color=["#B5BBC9", "#0B6E78"], width=0.55)
    ax.text(0, raw.mean() / 1024 + 0.5, f"{raw.mean() / 1024:.1f} KB", ha="center", fontsize=11, color=INK)
    ax.text(1, tx.mean() / 1024 + 0.5, f"{tx.mean() / 1024:.1f} KB", ha="center", fontsize=11, color=INK)
    ax.set_ylabel("downlink per scene (KB)"); ax.set_ylim(0, raw.mean() / 1024 * 1.2)
    ax.set_title(f"Bytes to downlink: {100 * (1 - tot_tx / tot_raw):.0f}% saved", fontsize=12, loc="left", color=INK, fontweight="bold")
    ax.text(0.5, -0.2, f"cloud is {cloud.mean() * 100:.0f}% of tiles, which caps the saving; unsure tiles are still sent",
            transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}: {n} tiles, cloud {cloud.mean():.1%}; cloud discarded rule {(dis_rule & cloud).sum() / cloud.sum():.1%} "
          f"mlp {(dis_mlp & cloud).sum() / cloud.sum():.1%}; clear lost rule {lost[0]:.1f}% mlp {lost[1]:.1f}%; bytes saved {1 - tot_tx / tot_raw:.1%}")


if __name__ == "__main__":
    main()
