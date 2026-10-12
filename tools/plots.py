"""Render a recipe's benchmark charts into <recipe>/img from <recipe>/results.json.

    uvx --with matplotlib python tools/plots.py recipes/countdown-qwen3-0.6b

Every chart is written in light and dark versions, as SVG (README, website) and PNG (Hugging Face, PRs).
Numbers live only in results.json, so a chart can never disagree with the table it illustrates.
"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "recipes/countdown-qwen3-0.6b")
R = json.loads((HERE / "results.json").read_text())
THEMES = {  # validated palette: blue/orange pass CVD + contrast checks on both surfaces
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  blue="#2a78d6", orange="#eb6834", base="#9a9993"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835",
                 blue="#3987e5", orange="#d95926", base="#7d7c76"),
}


def figure(t, w=7.2, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["ink2"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8); ax.set_axisbelow(True)
    return fig, ax


def title(fig, ax, t, text, sub):
    ax.set_title(text, loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=26)
    ax.text(0, 1.04, sub, transform=ax.transAxes, color=t["ink2"], fontsize=9)


def save(fig, name, theme):
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(HERE / "img" / f"{name}-{theme}.{ext}", facecolor=fig.get_facecolor())
    plt.close(fig)


def benchmarks(t, theme):
    fig, ax = figure(t)
    rows, w = R["benchmarks"], 0.36
    for i, b in enumerate(rows):
        for dx, key, color in ((-w / 2, "base", t["base"]), (w / 2, "trained", t["blue"])):
            ax.bar(i + dx, b[key], w - 0.04, color=color, zorder=2)  # gap between the pair
            ax.text(i + dx, b[key] + 1.5, f"{b[key]:.1f}%", ha="center", color=t["ink"], fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{b['name']}\n{b['note']}" for b in rows], color=t["ink2"], fontsize=8.5)
    ax.set_ylim(0, 122); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.bar(0, 0, color=t["base"], label="Qwen3-0.6B (base)"); ax.bar(0, 0, color=t["blue"], label="+ crucible_rl GRPO adapter")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=t["ink2"], ncol=2)
    title(fig, ax, t, "Held-out accuracy, base vs. RL-trained", "Greedy decoding, thinking on. Trained on easy Countdown only.")
    save(fig, "benchmarks", theme)


def curve(t, theme):
    c = R["training_curve"]
    fig, ax = figure(t)
    for key, color, label in (("format_ok", t["orange"], "format OK"), ("solved", t["blue"], "solved")):
        ax.plot(c["step"], c[key], color=color, linewidth=2, marker="o", markersize=4.5, zorder=3, label=label)
    best = c["solved"].index(max(c["solved"]))
    ax.annotate(f"best checkpoint\nstep {c['step'][best]}: {c['solved'][best]}%", (c["step"][best], c["solved"][best]),
                xytext=(c["step"][best] + 6, 62), color=t["ink"], fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=0.8))
    ax.set_xlim(-3, 103); ax.set_ylim(0, 105); ax.set_xlabel("training step", color=t["ink2"], fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    title(fig, ax, t, "Easy Countdown: held-out score during training", "256 held-out tasks, evaluated every 10 steps. About 15 min to the best checkpoint on one A10G.")
    save(fig, "training-curve", theme)


def overtraining(t, theme):
    fig, ax = figure(t)
    for run, color, label in ((R["training_curve"], t["blue"], "100-step run (best kept)"),
                              (R["overtraining_run"], t["orange"], "150-step run (last kept)")):
        ax.plot(run["step"], run["solved"], color=color, linewidth=2, marker="o", markersize=4.5, zorder=3, label=label)
    ax.set_ylim(0, 105); ax.set_xlim(-3, 155); ax.set_xlabel("training step", color=t["ink2"], fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    title(fig, ax, t, "Why we keep the best checkpoint", "Held-out solved %. Once the task is nearly solved, further GRPO steps can make it worse.")
    save(fig, "overtraining", theme)


def speed(t, theme):
    fig, ax = figure(t, h=2.4)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for i, (name, color) in enumerate((("HF generate", t["base"]), ("vLLM", t["blue"]))):
        lo, hi = R["speed_seconds_per_step"][name]
        ax.barh(i, hi - lo, left=lo, height=0.42, color=color, zorder=2)
        ax.text(hi + 4, i, f"{lo}–{hi} s", va="center", color=t["ink"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["HF generate", "vLLM (--vllm)"], color=t["ink2"], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 330); ax.set_xlabel("seconds per training step", color=t["ink2"], fontsize=9)
    title(fig, ax, t, "Rollout speed", "Same settings: 64 rollouts per step, thinking on, 512 tokens, one A10G.")
    save(fig, "speed", theme)


if __name__ == "__main__":
    (HERE / "img").mkdir(exist_ok=True)
    for theme, t in THEMES.items():
        for chart in (benchmarks, curve, overtraining, speed):
            chart(t, theme)
    print("wrote", sorted(p.name for p in (HERE / "img").iterdir()))
