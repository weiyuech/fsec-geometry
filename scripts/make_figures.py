"""Regenerate the README figures from the published result tables and measured example.

From the repository root: python scripts/make_figures.py
For print: python scripts/make_figures.py --format pdf --out /tmp/fsec-print
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import sys

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa.fit import fit_four_quadrants, quadrant
INK = "#202b33"
NOTE = "#475561"
RULE = "#dce2e4"
REFERENCE = "#74818a"
TEAL = "#116675"
ORANGE = "#a14b2d"
SEED = "#8db9bd"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.2, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": REFERENCE, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": NOTE, "ytick.color": NOTE, "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4, "xtick.major.size": 3, "ytick.major.size": 3,
        "svg.fonttype": "none", "svg.hashsalt": "fsec-audit",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig, out, stem, description, fmt):
    metadata = {"Title": stem.replace("-", " ").capitalize()}
    if fmt == "svg":
        metadata.update(Description=description, Date=None)
    if fmt == "pdf":
        metadata.update(Subject=description, CreationDate=None, ModDate=None)
    fig.savefig(out / f"{stem}.{fmt}", format=fmt, metadata=metadata, dpi=220)
    plt.close(fig)


def grid(ax, axis):
    ax.grid(True, axis=axis, color=RULE, linewidth=0.65)


def panel_title(ax, label):
    ax.set_title(label, loc="left", fontsize=12, fontweight="semibold", pad=10)


def legend_mark(marker, edge, fill):
    return Line2D([], [], linestyle="none", marker=marker, markersize=5.7,
                  markeredgecolor=edge, markerfacecolor=fill)


def french_tense(example, out, fmt):
    """Trace each threshold crossing in the raw grid, including its axis intercepts."""
    with np.load(example, allow_pickle=False) as data:
        angles = data["angles_deg"]
        grid_values = np.median(data["l2"], axis=0)
    fits = fit_four_quadrants(angles, grid_values)
    fig, ax = plt.subplots(figsize=(7.15, 5.3))
    fig.subplots_adjust(left=.12, right=.80, top=.84, bottom=.14)
    limit = 1.33
    ax.add_patch(Rectangle((0, 0), limit, limit, fc="#edf5f3", ec="none", zorder=0))
    theta = np.linspace(0, 2*np.pi, 361)
    reference, = ax.plot(np.cos(theta), np.sin(theta), color=REFERENCE,
                         linestyle=(0,(3,3)), lw=1.4, label="Circle (p = 2)")
    for signed, direction in [("++",(1,1)), ("+-",(1,-1)), ("-+",(-1,1)), ("--",(-1,-1))]:
        tense, french = direction
        a, g = quadrant(angles, grid_values, tense, french)
        fit = fits[signed]
        # This panel uses the same sign-specific sine normalization as the fitter.
        # Raw arrays index tense first, French second; French is horizontal here.
        lines = contourpy.contour_generator(a, a, g.T).lines(fits["thr"])
        for line in lines:
            radius = np.linalg.norm(line, axis=1)
            scale = np.pi / 180 * np.sinc(radius / 180)
            x = french * line[:,1] * scale / np.sin(np.deg2rad(fit["t2"]))
            y = tense * line[:,0] * scale / np.sin(np.deg2rad(fit["t1"]))
            ax.plot(x, y, color=TEAL, lw=2.3, solid_capstyle="round", zorder=3)
        ax.text(french*.42, tense*.41, f"p = {fit['p']:.2f}", ha="center", va="center", fontsize=10.5)
    ax.axvline(0, color=RULE, lw=1)
    ax.axhline(0, color=RULE, lw=1)
    ax.scatter([0],[0],s=12,color=INK,zorder=3)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal",
           xticks=[-1,0,1],yticks=[-1,0,1])
    ax.set_xlabel("← Reverse French                         Toward French →")
    ax.set_ylabel("← Past tense                               Future tense →")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    fig.text(.055,.944,"An equal-response contour in Gemma-2-9B",fontsize=14,fontweight="semibold")
    fig.text(.055,.891,"French × tense · median response over 30 text anchors", fontsize=10.5, color=NOTE)
    fig.legend([Line2D([],[],color=TEAL,lw=2.3), reference],
               ["Measured\nboundary","Circle (p = 2)"], bbox_to_anchor=(.735,.66), loc="upper left",
               frameon=False,fontsize=10,labelspacing=1.15,handlelength=2,handletextpad=.65)
    fig.text(.735,.40,"Shaded quadrant",fontsize=10.2,fontweight="semibold")
    fig.text(.735,.355,"The original sweep used\npositive French and\npositive tense.",fontsize=9.5,
             color=NOTE,va="top",linespacing=1.45)
    fig.text(.055,.02,"Every axis intercept is scaled to one unit. Each label gives the exponent fitted in that quadrant.",
             fontsize=9.2, color=NOTE)
    save(fig,out,"french-tense-boundary",
         "Measured contour of 30 median-pooled text anchors in Gemma-2-9B. French increases rightward, tense toward the future upward. The circle references p = 2; shading marks the original positive-positive sweep.",fmt)


def replication(results, out, fmt):
    with (results / "replication_pairs.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    old = np.array([float(row["upstream_p"]) for row in pairs])
    new = np.array([float(row["audit_p"]) for row in pairs])
    jobs = json.loads((results / "h4_controls.json").read_text())["jobs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.7, 4.35), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=.095, right=.98, bottom=.205, top=.80, wspace=.47)
    panel_title(a, "Same pairs, new anchors")
    a.scatter(old, new, s=11, color=TEAL, alpha=.49, linewidths=0, zorder=2)
    lo = np.floor(min(old.min(), new.min()) * 2) / 2
    hi = np.ceil(max(old.max(), new.max()) * 2) / 2
    a.plot([lo, hi], [lo, hi], color=REFERENCE, linestyle=(0, (3, 3)), linewidth=1, zorder=1)
    a.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Exponent p from published grids",
          ylabel="Exponent p from new grids", xticks=[2,3,4,5], yticks=[2,3,4,5], aspect="equal")
    a.text(.04, .97, f"{len(pairs)} pairs\nSpearman ρ = {spearmanr(old, new).statistic:.2f}",
           va="top", transform=a.transAxes, fontsize=9.5)
    grid(a, "both")
    panel_title(b, "Matched checkpoint controls")
    sizes = ["70m", "160m", "410m", "1b", "1.4b"]
    labels = ["Pythia 70M", "160M", "410M", "1B", "1.4B", "Gemma-2-9B"]

    def p(job):
        return jobs[job]["summary"]["ALL"]["median"]["p_paper"]

    for i, size in enumerate(sizes):
        trained = p(f"null_pythia-{size}_signed")
        untrained = p(f"null_pythia-{size}-step0_signed")
        b.plot([untrained, trained], [i,i], color=RULE, lw=2)
        b.plot(untrained, i, "o", mec=NOTE, mfc="white", ms=5.6)
        b.plot(trained, i, "o", color=TEAL, ms=6)
    y = len(sizes) + .3
    trained = p("g9_contrastive_signed")
    shuffles = [p(f"g9_shuffled{s}_signed") for s in range(3)]
    b.plot([min(shuffles), trained], [y,y], color=RULE, linewidth=2)
    b.plot(trained, y, "o", color=TEAL, ms=6)
    for offset, value in zip([-.19,0,.19], shuffles):
        b.plot(value, y+offset, "D", ms=4.5, mfc="white", mec=NOTE, mew=1)
    b.axvline(2, color=REFERENCE, linestyle=(0,(3,3)), linewidth=.9, zorder=0)
    b.set(yticks=[0,1,2,3,4,y], yticklabels=labels, ylim=(y+.6,-.6),
          xlim=(1.9,2.65), xticks=[2,2.2,2.4,2.6], xlabel="Median p over the matched 90 pairs")
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    grid(b, "x")
    handles = [legend_mark("o",TEAL,TEAL), legend_mark("o", NOTE,"white"), legend_mark("D",NOTE,"white")]
    fig.legend(handles, ["Trained", "Step zero", "Weight shuffle"], bbox_to_anchor=(.53,.995),
               loc="upper left", ncol=3, frameon=False, fontsize=9.3,
               handlelength=.9, columnspacing=1.15, handletextpad=.4)
    fig.text(.095, .045, f"Positive-positive quadrant. Gemma-2-9B medians on all {len(pairs)} pairs: "
             f"{np.median(old):.3f} in published grids and {np.median(new):.3f} in new grids.",
             fontsize=9.4, color=NOTE)
    save(fig,out,"replication-and-controls",
         "Matched pair exponents in published and new Gemma-2-9B grids, plus medians for trained, step-zero, and shuffled checkpoints.",fmt)


def language(results, out, fmt):
    data = json.loads((results / "gpu_shape.json").read_text())["g9_contrastive_signed"]
    figures = {}
    for family in ["lang","code"]:
        sign = data["sign_split"][family]
        directions = sign["by_direction"]
        fits = []
        for row in data["rows"]:
            if row["family"] != f"{family}-sem":
                continue
            target_axis = 0 if row["pair"][0] in directions else 1
            q = row["quadrants"]
            away, toward = [np.mean([v["p"] for name, v in q.items() if name[target_axis]==s])
                            for s in ("-","+")]
            fits.append((away, toward))
        figures[family] = np.array(fits)
    fig, (a,b) = plt.subplots(1,2,figsize=(8.7,4.85),gridspec_kw={"width_ratios":[1.25,.96]})
    fig.subplots_adjust(left=.09,right=.97,top=.83,bottom=.2,wspace=.46)
    panel_title(a, "Opposite signs of the same direction")
    lo, hi = 1.65, 3.5
    a.plot([lo,hi],[lo,hi],color=REFERENCE,linestyle=(0,(3,3)),linewidth=1)
    handles=[]
    for family, color, marker, label in [("code",ORANGE,"^","Code"),("lang",TEAL,"o","Language")]:
        away, toward = figures[family].T
        a.scatter(away,toward,s=14,marker=marker,color=color,alpha=.62,linewidths=0)
        handles.append(legend_mark(marker,color,color))
    a.set(xlim=(lo,hi), ylim=(lo,hi), xticks=[2,2.5,3,3.5], yticks=[2,2.5,3,3.5], aspect="equal",
          xlabel="Exponent p in reverse", ylabel="Exponent p toward target")
    grid(a,"both")
    fig.legend(handles[::-1], [f"Language × semantic ({len(figures['lang'])})",f"Code × semantic ({len(figures['code'])})"],
               loc="upper left", bbox_to_anchor=(.07,.966), ncol=2,
               frameon=False, handlelength=1, handletextpad=.4, columnspacing=1.8, fontsize=9.7)
    panel_title(b, "Difference by language")
    sign = data["sign_split"]["lang"]
    ordered = sorted(sign["by_direction"].items(),key=lambda kv: -kv[1]["median_p_difference"])
    yy = np.arange(len(ordered))
    medians = [s["median_p_difference"] for _,s in ordered]
    b.hlines(yy,0,medians,color=RULE,linewidth=1.7)
    b.scatter(medians, yy, color=TEAL, s=29,zorder=3)
    b.set(yticks=yy,yticklabels=[name for name,_ in ordered],ylim=(yy[-1]+.6,-.6),
          xlim=(0,1.04),xticks=[0,.25,.5,.75,1],
          xlabel="Median paired change in p\n(toward target − reverse)")
    b.tick_params(axis="y",length=0)
    b.spines["left"].set_visible(False)
    grid(b,"x")
    count = sign["count_p_plus_greater"]
    fig.text(.09,.045, f"Across {count} language pairs, median paired change = "
             f"+{sign['p_difference'][0]:.3f}. Each sign averages over both semantic signs.",
             fontsize=9.1,color=NOTE)
    save(fig,out,"language-sign",
         "Toward-target and reverse exponents for individual language and code pairs, plus median paired shifts for ten languages in Gemma-2-9B.",fmt)


def toy(results, out, fmt):
    with (results / "toy_pairs.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if float(r["amax"]) == 25 and float(r["theta"]) == 0]
    dims = sorted({int(r["d"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    noises = sorted({float(r["anchor_noise"]) for r in rows})
    values = {(int(r["d"]),int(r["seed"]),float(r["anchor_noise"])):float(r["p_median"]) for r in rows}
    if len(values) != len(rows) or len(values) != len(dims)*len(seeds)*len(noises):
        raise ValueError("Expected one fitted median per model dimension, seed and noise level")
    fig, axes = plt.subplots(1,len(dims),sharex=True,sharey=True,figsize=(8.7,3.75))
    fig.subplots_adjust(left=.09,right=.985,top=.79,bottom=.24,wspace=.19)
    for ax,d in zip(axes,dims):
        by_seed = [[values[d,s,n] for n in noises] for s in seeds]
        ax.axhline(2,color=REFERENCE,linestyle=(0,(3,3)),linewidth=1.15,zorder=1)
        for ys in by_seed:
            ax.plot(noises,ys,color=SEED,linewidth=.9,marker=".",markersize=4,zorder=2)
        ax.plot(noises,[median(v) for v in zip(*by_seed)],color=TEAL,linewidth=2,marker="o",markersize=4.5)
        ax.set(xlim=(-.009,max(noises)+.009),ylim=(1.5,3.5),xticks=noises,
               xticklabels=[f"{n:.4f}" if n else "0" for n in noises],yticks=[1.5,2,2.5,3,3.5])
        panel_title(ax,f"{d:,} features")
        grid(ax,"y")
    axes[0].set_ylabel("Fitted exponent p")
    fig.text(.54,.105,"Anchor noise (standard deviation)",ha="center",fontsize=10.3)
    fig.legend([Line2D([],[],color=TEAL,marker="o",lw=2,ms=4),
                Line2D([],[],color=SEED,marker=".",lw=1),
                Line2D([],[],color=REFERENCE,linestyle=(0,(3,3)),lw=1)],
               ["Median of five seeds","Individual seed","Circular boundary (p = 2)"],
               loc="upper left",bbox_to_anchor=(.075,.976),ncol=3,frameon=False,fontsize=9.6,columnspacing=1.85)
    fig.text(.09,.023,"Probes aligned with the true feature axes. Maximum perturbation 25. "
             "The toy's design noise is 0.1732.",fontsize=9.3,color=NOTE)
    save(fig,out,"toy-anchor-noise",
         "Fitted pair exponent versus anchor noise for 4,096, 8,192 and 16,384-dimensional toy denoisers. Each panel shows five seeds and their median.",fmt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT/"results")
    p.add_argument("--example", type=Path, default=ROOT/"examples/Tense__French.npz")
    p.add_argument("--out",type=Path,default=ROOT/"figures")
    p.add_argument("--format",choices=["svg","pdf","png"],default="png")
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    style()
    french_tense(args.example,args.out,args.format)
    for make in (replication,language,toy):
        make(args.results,args.out,args.format)


if __name__ == "__main__":
    main()
