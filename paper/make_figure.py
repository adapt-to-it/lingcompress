"""Figure 1: GSM8K accuracy vs generated tokens per format, one panel per model size."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
# TeX Gyre Termes as in the paper (TeX Live path on Linux); falls back to another Times-like serif if missing.
for f in ("regular","bold","italic"):
    p = Path(f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{f}.otf")
    if p.exists():
        font_manager.fontManager.addfont(str(p))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "pdf.fonttype": 42,
})

# format: (color, marker, label)
STYLE = {
    "A":  ("#2a78d6", "o", "A full prose"),
    "A1": ("#eb6834", "D", "A1 concise prose"),
    "A2": ("#1baf7a", "^", "A2 very concise"),
    "C":  ("#eda100", "s", "C canonical"),
}

# (mean generated tokens, accuracy %), all with calculator; C also grammar-constrained
DATA = {
    "Qwen3-0.6B": {"A": (101.1, 57.6), "A1": (75.1, 59.2), "A2": (64.4, 56.0), "C": (75.9, 53.7)},
    "Qwen3-1.7B": {"A": (101.6, 72.5), "A1": (76.5, 71.2), "A2": (65.3, 69.1), "C": (74.1, 69.1)},
    "Qwen3-4B (QLoRA)": {"A": (102.9, 82.9), "A1": (75.8, 82.6), "C": (75.5, 79.4)},
}

# label offsets in points, chosen so labels never collide with each other
OFFSETS = {"A": (-4, 6, "right"), "A1": (6, 3, "left"), "A2": (5, -9, "left"), "C": (6, -9, "left")}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05))
for ax, (model, pts) in zip(axes, DATA.items()):
    ys = [p[1] for p in pts.values()]
    for fmt, (tok, acc) in pts.items():
        color, marker, _ = STYLE[fmt]
        ax.scatter(tok, acc, s=34, color=color, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=3)
        dx, dy, ha = OFFSETS[fmt]
        ax.annotate(fmt, (tok, acc), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8.5, color="#0b0b0b")
    ax.set_title(model, fontsize=9.5, pad=4)
    ax.set_xlim(58, 110)
    ax.set_ylim(min(ys) - 3, max(ys) + 3)
    ax.grid(True, color="#e6e5e1", linewidth=0.5, zorder=0)
    ax.tick_params(length=2, width=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Generated tokens (mean)")
axes[0].set_ylabel("Accuracy (%)")

handles = [plt.Line2D([], [], color=c, marker=m, linestyle="", markersize=5,
                      markeredgecolor="white", label=l) for c, m, l in STYLE.values()]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.1), fontsize=8.5, handletextpad=0.3, columnspacing=1.4)
fig.tight_layout(w_pad=1.2)
fig.savefig("fig_frontier.pdf", bbox_inches="tight")
fig.savefig("fig_frontier.png", dpi=200, bbox_inches="tight")
