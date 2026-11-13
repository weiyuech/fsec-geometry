"""Draw the README preview from the measured contour used by the interactive figure."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from export_explorer import export

ROOT = Path(__file__).resolve().parents[1]
PAPER = "#fbfaf7"
INK = "#263537"
MUTED = "#56666a"
TEAL = "#176875"
COPPER = "#9a4b34"
AXIS = "#bcc4c0"
REFERENCE = "#94a3a0"
STATES = (
    ("topRight", "Toward French", "Future tense", 45),
    ("topLeft", "Opposite French", "Future tense", 135),
    ("bottomLeft", "Opposite French", "Past tense", 225),
    ("bottomRight", "Toward French", "Past tense", 315),
)


def make_scene(data):
    # 2.4× the intended 500 CSS-pixel width. Equal aspect preserves measured distances.
    fig = plt.figure(figsize=(5, 3.12), dpi=240, facecolor=PAPER)
    ax = fig.add_axes([0.046, 0.117, 0.514, 0.777], aspect="equal")
    ax.set(xlim=(-1.34, 1.34), ylim=(-1.34, 1.34))
    ax.axis("off")
    circle = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(circle), np.sin(circle), color=REFERENCE, linewidth=0.9,
            dashes=(2.2, 2.2), solid_capstyle="round")
    ax.plot([-1.22, 1.22], [0, 0], color=AXIS, linewidth=0.65, zorder=0)
    ax.plot([0, 0], [-1.20, 1.20], color=AXIS, linewidth=0.65, zorder=0)
    for quarter in data["quadrants"].values():
        points = np.asarray(quarter["points"])
        ax.plot(*points.T, color=TEAL, linewidth=1.25, solid_capstyle="round")
    ax.plot(0, 0, "o", color=INK, markeredgecolor=PAPER, markeredgewidth=0.6, ms=4.1, zorder=6)
    ax.annotate("0", (0, 0), xytext=(-7, -10.5), textcoords="offset points",
                color=MUTED, fontsize=7.6, zorder=7)
    for x, y, label, ha, va in (
        (0, 1.255, "Future", "center", "center"),
        (0, -1.255, "Past", "center", "center"),
        (1.15, -0.07, "+French", "left", "top"),
        (-1.105, -0.07, "−French", "right", "top"),
    ):
        ax.text(x, y, label, ha=ha, va=va, size=7.8, color=MUTED, clip_on=False)

    left = 0.643
    fig.text(left, 0.862, "French × tense", family="DejaVu Serif", size=13.3, color=INK)
    state = fig.text(left, 0.700, "", size=10, color=INK, linespacing=1.4)
    fig.text(left, 0.496, "Boundary distance", size=8.6, color=MUTED)
    distance = fig.text(left, 0.360, "", size=22, family="DejaVu Serif", color=COPPER)
    fig.text(left, 0.276, "Single-axis distance = 1", size=8.1, color=MUTED)
    for y, color, dash, label in ((0.163, TEAL, None, "Measured boundary"),
                                  (0.084, REFERENCE, (2.2, 2.2), "Unit circle")):
        sample = plt.Line2D([left, left + 0.042], [y, y], color=color, linewidth=1.2,
                            transform=fig.transFigure, solid_capstyle="round")
        if dash:
            sample.set_dashes(dash)
        fig.add_artist(sample)
        fig.text(left + 0.060, y, label, size=8.0, va="center", color=MUTED)
    shaft, = ax.plot([], [], color=COPPER, linewidth=1.45, solid_capstyle="round", zorder=3)
    arrowhead = Polygon(np.zeros((3, 2)), facecolor=COPPER, edgecolor="none", zorder=4)
    ax.add_patch(arrowhead)
    return fig, state, distance, shaft, arrowhead


def render(scene, data, spec, fraction):
    fig, state, distance, shaft, arrowhead = scene
    key, french, tense, angle = spec
    q = data["quadrants"][key]
    measured = np.asarray(q["diagonalPoint"])
    radial = [data["raySamples"][angle][k] for k in ("x", "y")]
    if not np.allclose(measured, radial, atol=1e-8):
        raise ValueError(f"Diagonal endpoint differs from measured ray in {key}")
    # Interpolate only the illustrated push; keep the measured boundary fixed.
    end = fraction * measured
    unit = measured / np.linalg.norm(measured)
    normal = np.array([-unit[1], unit[0]])
    length = min(0.085, np.linalg.norm(end) * 0.5)
    base = end - length * unit
    shaft.set_data([0, base[0]], [0, base[1]])
    arrowhead.set_xy([end, base + 0.45 * length * normal, base - 0.45 * length * normal])
    state.set_text(f"{french}\n{tense}")
    distance.set_text(f"{q['diagonalRadius']:.2f}")
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/quadrant-preview.gif")
    path = parser.parse_args().out
    data = export()
    scene = make_scene(data)
    palette = None
    frames, delays = [], []
    # Hold the measured endpoint. Retract to zero, change one sign, then extend.
    # Cosine easing adjusts only timing along the straight ray.
    travel = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 9))
    for i, current in enumerate(STATES):
        following = STATES[(i + 1) % len(STATES)]
        samples = [(current, 1.0, 1500)]
        samples += [(current, f, 50) for f in travel[-2:0:-1]]
        samples += [(following, 0.0, 70)]
        samples += [(following, f, 50) for f in travel[1:-1]]
        for state, fraction, delay in samples:
            image = render(scene, data, state, fraction)
            if palette is None:
                palette = image.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
            frames.append(image.quantize(palette=palette, dither=Image.Dither.NONE))
            delays.append(delay)
    plt.close(scene[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=delays,
                   loop=0, optimize=True)
    print(f"{path.name}: {len(frames)} frames, {sum(delays)/1000:.2f} seconds")


if __name__ == "__main__":
    main()
