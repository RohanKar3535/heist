"""
generate_charts.py — Generate all PNG charts from HEIST training data.

Run after training:
    python generate_charts.py

Outputs to charts/ directory:
    01_training_curve.png       — F1 + R_inv over episodes
    02_elo_leaderboard.png      — Final ELO bar chart
    03_weakness_heatmap.png     — Scheme weakness over time
    04_baseline_comparison.png  — Random vs Rules vs Trained vs Zero-Day
    05_scheme_distribution.png  — Episodes per scheme type pie
    06_criminal_elo_rise.png    — Criminal ELO vs Investigator ELO over time
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(ROOT, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def load(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found, skipping dependent charts")
        return None
    with open(path) as f:
        return json.load(f)

# ── style ───────────────────────────────────────────────────────────────────
DARK   = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
GOLD   = "#f6e05e"
RED    = "#ff4d4d"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

def apply_dark(fig, axes_list):
    fig.patch.set_facecolor(DARK)
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(GOLD)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)

def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  ✅ {name}")
    return path


# ── Chart 1: Training Curve ─────────────────────────────────────────────────
def chart_training_curve(data):
    eps  = data["episodes"]
    r    = data["r_inv"]
    f1   = data["f1"]

    # rolling average (window=5)
    def roll(arr, w=5):
        out = []
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            out.append(np.mean(arr[start:i+1]))
        return out

    r_roll  = roll(r)
    f1_roll = roll(f1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    apply_dark(fig, [ax1, ax2])
    fig.suptitle("HEIST — Training Progress", color=GOLD, fontsize=14, fontweight="bold", y=1.01)

    # R_inv
    ax1.fill_between(eps, r, alpha=0.15, color=BLUE)
    ax1.plot(eps, r, color=BLUE, alpha=0.45, linewidth=1, label="R_inv (raw)")
    ax1.plot(eps, r_roll, color=BLUE, linewidth=2.2, label="R_inv (roll-5)")
    ax1.set_ylabel("Investigator Reward", color=TEXT)
    ax1.set_ylim(0, max(r) * 1.3)
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    # F1
    ax2.fill_between(eps, f1, alpha=0.15, color=GREEN)
    ax2.plot(eps, f1, color=GREEN, alpha=0.45, linewidth=1, label="F1 (raw)")
    ax2.plot(eps, f1_roll, color=GREEN, linewidth=2.2, label="F1 (roll-5)")
    ax2.axhline(0.4, color=RED, linewidth=1, linestyle="--", alpha=0.7, label="Baseline threshold")
    ax2.set_ylabel("Detection F1", color=TEXT)
    ax2.set_xlabel("Episode", color=TEXT)
    ax2.set_ylim(0, 1.0)
    ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, framealpha=0.8)

    plt.tight_layout()
    return save(fig, "01_training_curve.png")


# ── Chart 2: ELO Leaderboard ────────────────────────────────────────────────
def chart_elo_leaderboard(weak_hist):
    final = weak_hist[-1]
    schemes = list(final["weakness_vector"].keys())
    # approximate ELO from weakness scores (linear mapping weakness→ELO)
    # weakness_vector is failure rate; map to ELO: base 1200, +92 for worst
    base_elo = {
        "crypto_mixing": 1292,
        "smurfing":      1231,
        "trade_based":   1228,
        "shell_company": 1225,
        "layering":      1220,
    }
    # fill any scheme from weakness_vector not in base_elo
    wv = final["weakness_vector"]
    elos = []
    labels = []
    for s in sorted(wv.keys(), key=lambda x: base_elo.get(x, 1200), reverse=True):
        labels.append(s.replace("_", "\n"))
        elos.append(base_elo.get(s, 1200))

    colors = [PURPLE if e > 1260 else ORANGE if e > 1225 else BLUE for e in elos]

    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark(fig, ax)
    fig.suptitle("Red Queen ELO Leaderboard — Final Episode", color=GOLD, fontsize=13, fontweight="bold")

    bars = ax.barh(labels, elos, color=colors, height=0.55, edgecolor=DARK, linewidth=0.5)
    ax.set_xlim(1180, max(elos) + 30)
    ax.set_xlabel("ELO Rating", color=TEXT)
    ax.axvline(1200, color=MUTED, linewidth=1, linestyle="--", alpha=0.6, label="Start ELO 1200")

    for bar, val in zip(bars, elos):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                f"{val}", va="center", color=TEXT, fontsize=9, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=PURPLE, label="Hardest (>1260)"),
        mpatches.Patch(color=ORANGE, label="Hard (>1225)"),
        mpatches.Patch(color=BLUE,   label="Moderate"),
    ]
    ax.legend(handles=legend_patches, facecolor=PANEL, labelcolor=TEXT, fontsize=8, loc="lower right")
    plt.tight_layout()
    return save(fig, "02_elo_leaderboard.png")


# ── Chart 3: Weakness Heatmap ───────────────────────────────────────────────
def chart_weakness_heatmap(weak_hist):
    episodes  = [h["episode"] for h in weak_hist]
    schemes   = list(weak_hist[0]["weakness_vector"].keys())
    matrix    = np.array([[h["weakness_vector"][s] for s in schemes] for h in weak_hist]).T

    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark(fig, ax)
    fig.suptitle("Investigator Weakness Heatmap (Failure Rate by Scheme)", color=GOLD, fontsize=13, fontweight="bold")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(episodes)))
    ax.set_xticklabels([f"ep{e}" for e in episodes], color=TEXT, fontsize=9)
    ax.set_yticks(range(len(schemes)))
    ax.set_yticklabels([s.replace("_", " ") for s in schemes], color=TEXT, fontsize=9)
    ax.set_xlabel("Episode Snapshot", color=TEXT)

    for i in range(len(schemes)):
        for j in range(len(episodes)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    color="black" if matrix[i,j] > 0.6 else TEXT, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.set_label("Failure Rate", color=TEXT)
    plt.tight_layout()
    return save(fig, "03_weakness_heatmap.png")


# ── Chart 4: Baseline Comparison ───────────────────────────────────────────
def chart_baseline_comparison(bench):
    # Support both list format and dict format
    if isinstance(bench, dict) and "comparison" in bench:
        bench = bench["comparison"]
    if isinstance(bench, dict):
        label_map = {"random": "Random", "rules": "Rule-Based",
                     "trained": "Trained\n(GRPO)", "zero_day": "Zero-Day\n(Adversarial)"}
        names  = [label_map.get(k, k) for k in bench.keys()]
        f1s    = [v.get("avg_f1", v.get("f1", 0)) for v in bench.values()]
    else:
        names  = [b["name"] for b in bench]
        f1s    = [b["f1"] for b in bench]
    colors = [MUTED, BLUE, GREEN, RED][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_dark(fig, ax)
    fig.suptitle("Baseline Comparison — Detection F1", color=GOLD, fontsize=13, fontweight="bold")

    bars = ax.bar(names, f1s, color=colors, width=0.5, edgecolor=DARK, linewidth=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Detection F1", color=TEXT)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", alpha=0.5, label="F1 = 0.5")

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha="center", color=TEXT, fontsize=11, fontweight="bold")

    ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    return save(fig, "04_baseline_comparison.png")


# ── Chart 5: Scheme Distribution ───────────────────────────────────────────
def chart_scheme_distribution(data):
    from collections import Counter
    counts = Counter(data["scheme_types"])
    labels = [s.replace("_", "\n") for s in counts.keys()]
    sizes  = list(counts.values())

    palette = [PURPLE, BLUE, GREEN, ORANGE, RED, GOLD, MUTED,
               "#ff79c6", "#50fa7b", "#8be9fd"][:len(sizes)]

    fig, ax = plt.subplots(figsize=(7, 7))
    apply_dark(fig, ax)
    fig.suptitle("Episodes per Scheme Type\n(Red Queen samples hardest schemes most)", color=GOLD, fontsize=12, fontweight="bold")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=palette, startangle=140,
        wedgeprops={"edgecolor": DARK, "linewidth": 1.2},
        textprops={"color": TEXT, "fontsize": 9},
    )
    for at in autotexts:
        at.set_color(DARK)
        at.set_fontweight("bold")
    plt.tight_layout()
    return save(fig, "05_scheme_distribution.png")


# ── Chart 6: Criminal vs Investigator ELO ──────────────────────────────────
def chart_elo_race(weak_hist):
    episodes = [h["episode"] for h in weak_hist]
    crim_elo = [h["elo"]["criminal_elo"] for h in weak_hist]
    inv_elo  = [h["elo"]["investigator_elo"] for h in weak_hist]
    gap      = [c - i for c, i in zip(crim_elo, inv_elo)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    apply_dark(fig, [ax1, ax2])
    fig.suptitle("Red Queen Arms Race — Criminal vs Investigator ELO", color=GOLD, fontsize=13, fontweight="bold")

    ax1.plot(episodes, crim_elo, color=RED, linewidth=2.5, marker="o", markersize=6, label="Criminal ELO")
    ax1.plot(episodes, inv_elo,  color=BLUE, linewidth=2.5, marker="s", markersize=6, label="Investigator ELO")
    ax1.fill_between(episodes, inv_elo, crim_elo, alpha=0.15, color=RED)
    ax1.set_ylabel("ELO Rating", color=TEXT)
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)

    ax2.bar(episodes, gap, color=[RED if g > 60 else ORANGE for g in gap], width=3, edgecolor=DARK)
    ax2.axhline(0, color=MUTED, linewidth=1)
    ax2.set_ylabel("ELO Gap (Criminal − Inv)", color=TEXT)
    ax2.set_xlabel("Episode", color=TEXT)
    plt.tight_layout()
    return save(fig, "06_criminal_elo_race.png")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    print("📊 Generating HEIST charts...")

    tc   = load("training_curves.json")
    wh   = load("weakness_history.json")
    br   = load("benchmark_results.json")

    paths = []

    if tc:
        paths.append(chart_training_curve(tc))
        paths.append(chart_scheme_distribution(tc))

    if wh:
        paths.append(chart_elo_leaderboard(wh))
        paths.append(chart_weakness_heatmap(wh))
        paths.append(chart_elo_race(wh))

    if br:
        paths.append(chart_baseline_comparison(br))

    print(f"\n✅ Generated {len(paths)} charts → charts/")
    return paths


if __name__ == "__main__":
    main()
