"""Nature-style result figures for the hydrogen-store service manuscript.

The script reads only the audited result tables of the completed model batch and
re-draws the four result figures. No number is recomputed, re-derived or edited:
every plotted value is taken from a recorded CSV column. Outputs are vector
(PDF, SVG, EPS) plus 1200 dpi raster copies.

A layout audit runs inside the figure build and fails the build if
  * two text items overlap,
  * a text box is crossed by a plotted line, marker or scatter point,
  * a text box leaves the canvas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
MAIN = RESULTS / "eho_site_module_frontier" / "primary_cyclic_v5_ds1_ml06_ml04_b24_holdout2015_2023"
FRONTIER = RESULTS / "eho_threshold_sensitivity" / "cyclic_continuous_module_frontier_v1"
RATE = RESULTS / "eho_site_module_frontier" / "declared_anchor_subgrid_v1"
OUT = Path(__file__).resolve().parents[1] / "figures"

CASES = ["rigid", "flex_ml0p6_b24", "flex_ml0p4_b24"]
LABELS = {
    "rigid": "Rigid service",
    "flex_ml0p6_b24": "60% floor, 24 h buffer",
    "flex_ml0p4_b24": "40% floor, 24 h buffer",
}
COLORS = {"rigid": "#4D4D4D", "flex_ml0p6_b24": "#0072B2", "flex_ml0p4_b24": "#D55E00"}
MARKERS = {"rigid": "o", "flex_ml0p6_b24": "s", "flex_ml0p4_b24": "^"}
ANCHORS = ["spain_sabinanigo", "spain_huelva", "netherlands_sluiskil"]
ANCHOR_LABEL = {
    "spain_sabinanigo": "Sabi\u00f1\u00e1nigo",
    "spain_huelva": "Huelva",
    "netherlands_sluiskil": "Sluiskil",
}
SITE_LABEL = {
    "france_grand_quevilly": "Grand Quevilly",
    "france_grandpuits_bailly_carrois": "Grandpuits Bailly Carrois",
    "france_le_havre": "Le Havre",
    "france_ottmarsheim": "Ottmarsheim",
    "italy_ferrara": "Ferrara",
    "netherlands_geleen": "Geleen",
    "netherlands_sluiskil": "Sluiskil",
    "poland_kedzierzyn": "K\u0119dzierzyn",
    "poland_police": "Police",
    "poland_pulawy": "Pu\u0142awy",
    "poland_tarnow": "Tarn\u00f3w",
    "poland_wloclawek": "W\u0142oc\u0142awek",
    "spain_huelva": "Huelva",
    "spain_puertollano": "Puertollano",
    "spain_sabinanigo": "Sabi\u00f1\u00e1nigo",
}
YEARS = [2015, 2019, 2023]
WIDTH_MM = 185.0  # MDPI full text length (\fulllength)
WIDTH_NARROW_MM = 133.0  # single-panel figure width
AUDIT: list[dict] = []
STYLE_COMMIT = "b9b16959570bd2fbc9ff5118bacc423c3bddd592"


def mm(value: float) -> float:
    return value / 25.4


def style() -> None:
    vendor = SCRIPT_DIR / "vendor" / "scienceplots"
    # Actual pinned SciencePlots presets, not a hand-written imitation.
    plt.style.use([vendor / name for name in
                   ("science.mplstyle", "nature.mplstyle", "no-latex.mplstyle")])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "xtick.top": False,
            "ytick.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "axes.grid": False,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# --------------------------------------------------------------------------
# layout audit
# --------------------------------------------------------------------------
def collect_texts(fig: plt.Figure) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for ai, ax in enumerate(fig.axes):
        for ti, t in enumerate(ax.texts):
            items.append((f"ax{ai}.text{ti}", t))
        items.append((f"ax{ai}.xlabel", ax.xaxis.label))
        items.append((f"ax{ai}.ylabel", ax.yaxis.label))
        for loc, attr in (("left", "_left_title"), ("center", "title"), ("right", "_right_title")):
            artist = getattr(ax, attr, None)
            if artist is not None and artist.get_text().strip():
                items.append((f"ax{ai}.title.{loc}", artist))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for tick, t in zip(ax.xaxis.get_major_ticks(), ax.get_xticklabels(which="both")):
            if x0 <= tick.get_loc() <= x1:
                items.append((f"ax{ai}.xtick", t))
        for tick, t in zip(ax.yaxis.get_major_ticks(), ax.get_yticklabels(which="both")):
            if y0 <= tick.get_loc() <= y1:
                items.append((f"ax{ai}.ytick", t))
        legend = ax.get_legend()
        items.append((f"ax{ai}.xoffset", ax.xaxis.get_offset_text()))
        items.append((f"ax{ai}.yoffset", ax.yaxis.get_offset_text()))
        if legend is not None:
            for t in legend.get_texts():
                items.append((f"ax{ai}.legend", t))
    for li, legend in enumerate(fig.legends):
        for t in legend.get_texts():
            items.append((f"fig.legend{li}", t))
    for ti, t in enumerate(fig.texts):
        items.append((f"fig.text{ti}", t))
    return items


def boxes(items, renderer):
    out = []
    for tag, artist in items:
        if not artist.get_visible():
            continue
        try:
            text = artist.get_text()
        except Exception:
            continue
        if not text.strip():
            continue
        try:
            ext = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if ext.width <= 0 or ext.height <= 0:
            continue
        out.append((tag, text.replace("\n", " | "), Bbox.from_extents(ext.x0, ext.y0, ext.x1, ext.y1)))
    return out


def inflate(bb: Bbox, pad: float = 0.7) -> Bbox:
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def overlap(a: Bbox, b: Bbox) -> float:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return min(dx, dy)


def segment_hits_box(p0, p1, bb: Bbox) -> bool:
    """Liang-Barsky segment against axis aligned box test."""
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bb.x0), (dx, bb.x1 - x0), (-dy, y0 - bb.y0), (dy, bb.y1 - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


def data_polylines(fig: plt.Figure) -> list[tuple[str, list[tuple[float, float]]]]:
    out = []
    for ai, ax in enumerate(fig.axes):
        for li, line in enumerate(ax.lines):
            if not line.get_visible() or line.get_linestyle() in ("", "None", "none"):
                continue
            data = line.get_xydata()
            if data is None or len(data) == 0:
                continue
            pts = [tuple(map(float, p)) for p in ax.transData.transform(np.asarray(data, dtype=float))]
            if len(pts) == 1:
                pts = [pts[0], pts[0]]
            out.append((f"ax{ai}.line{li}", pts))
        for ci, coll in enumerate(ax.collections):
            if isinstance(coll, LineCollection) and coll.get_visible():
                for si, segment in enumerate(coll.get_segments()):
                    pts = [tuple(map(float, p)) for p in coll.get_transform().transform(segment)]
                    out.append((f"ax{ai}.linecollection{ci}.{si}", pts))
    return out


def data_points(fig: plt.Figure):
    out = []
    for ai, ax in enumerate(fig.axes):
        for ci, coll in enumerate(ax.collections):
            if not isinstance(coll, PathCollection):
                continue
            offsets = coll.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue
            pts = [tuple(map(float, p)) for p in ax.transData.transform(np.asarray(offsets, dtype=float))]
            radius = float(np.sqrt(np.max(coll.get_sizes()))) * fig.dpi / 72 / 2
            out.append((f"ax{ai}.points{ci}", pts, radius))
        for li, line in enumerate(ax.lines):
            if not line.get_visible() or line.get_marker() in ("", "None", "none"):
                continue
            pts = [tuple(map(float, p)) for p in line.get_transform().transform(line.get_xydata())]
            radius = line.get_markersize() * fig.dpi / 72 / 2
            out.append((f"ax{ai}.linemarkers{li}", pts, radius))
    return out


def audit_figure(fig: plt.Figure, name: str) -> dict:
    # Geometry is tested at 300 dpi, with clearance in physical points.
    fig.set_dpi(300)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    canvas = Bbox.from_extents(-1.0, -1.0, width + 1.0, height + 1.0)
    entries = boxes(collect_texts(fig), renderer)
    problems: list[str] = []
    clearance = 0.25 * fig.dpi / 72

    for i, (tag_i, text_i, bb_i) in enumerate(entries):
        if not (canvas.contains(bb_i.x0, bb_i.y0) and canvas.contains(bb_i.x1, bb_i.y1)):
            problems.append(f"outside canvas: {tag_i} '{text_i[:44]}'")
        for tag_j, text_j, bb_j in entries[i + 1:]:
            if overlap(inflate(bb_i, clearance), inflate(bb_j, clearance)) > 0:
                problems.append(f"text/text overlap: {tag_i} '{text_i[:36]}' vs {tag_j} '{text_j[:36]}'")

    lines = data_polylines(fig)
    points = data_points(fig)
    for tag, text, bb in entries:
        probe = inflate(bb, clearance)
        for ltag, pts in lines:
            if any(segment_hits_box(p0, p1, probe) for p0, p1 in zip(pts[:-1], pts[1:])):
                problems.append(f"text/line crossing: {tag} '{text[:36]}' vs {ltag}")
        for ptag, pts, radius in points:
            marker_probe = inflate(bb, radius + clearance)
            hit = [p for p in pts if marker_probe.contains(p[0], p[1])]
            if hit:
                problems.append(f"text/marker overlap: {tag} '{text[:36]}' vs {ptag} ({len(hit)} points)")
        for ai, ax in enumerate(fig.axes):
            for pi, patch in enumerate(ax.patches):
                if patch.get_visible() and overlap(probe, patch.get_window_extent(renderer)) > 0:
                    problems.append(f"text/bar overlap: {tag} '{text[:36]}' vs ax{ai}.patch{pi}")

    entry = {"figure": name, "text_items": len(entries), "problems": problems,
             "audit_dpi": 300, "clearance_pt": 0.25,
             "checks": ["text/text", "text/line", "text/line-collection",
                        "text/marker-footprint", "text/bar", "canvas-containment"]}
    AUDIT.append(entry)
    return entry


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entry = audit_figure(fig, name)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.eps")
    fig.savefig(OUT / f"{name}.png", dpi=1200)
    plt.close(fig)
    print(f"[{name}] texts={entry['text_items']} problems={len(entry['problems'])}")


def panel_letter(ax, letter: str, dx: float = 0.012, dy: float = 0.985) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def year_axis(ax) -> None:
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_xlabel("Weather year")
    ax.set_xlim(-0.55, len(YEARS) - 0.45)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figure1(frontier: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(mm(WIDTH_MM), mm(62)), sharey=True, constrained_layout=True)
    offsets = {case: 0.0 for case in CASES}
    for ax, sid in zip(axes, ANCHORS):
        for case in CASES:
            sub = frontier[(frontier.site_id == sid) & (frontier.process_case == case)].sort_values("weather_year")
            x = [YEARS.index(y) + offsets[case] for y in sub.weather_year]
            y = sub.maximum_service_tph_per_reference_module.to_numpy(dtype=float)
            ax.plot(x, y, color=COLORS[case], marker=MARKERS[case], lw=0.9, ms=3.6,
                    markerfacecolor=COLORS[case], markeredgecolor="white", markeredgewidth=0.5)
        year_axis(ax)
        ax.set_title(ANCHOR_LABEL[sid], loc="left")
        ax.set_ylim(0, 50)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
    axes[0].set_ylabel("Service capacity per reference unit\n(t NH$_3$-equivalent h$^{-1}$)")
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], lw=0.9, ms=3.6,
                      markerfacecolor=COLORS[c], markeredgecolor="white", markeredgewidth=0.5,
                      label=LABELS[c]) for c in CASES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handlelength=2.2)
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter, dx=0.015, dy=0.965)
    save(fig, "figure1_service_capacity_frontier")


def figure2(paired: pd.DataFrame, by_site: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(mm(WIDTH_MM), mm(79)), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax = axes[0]
    unchanged = paired.reference_caverns_avoided.eq(0)
    ax.scatter(paired.loc[unchanged, "free_h2_storage_service_hours_rigid"],
               paired.loc[unchanged, "free_h2_reduction_pct"],
               s=11, color="#8C8C8C", marker="o", linewidths=0)
    ax.scatter(paired.loc[~unchanged, "free_h2_storage_service_hours_rigid"],
               paired.loc[~unchanged, "free_h2_reduction_pct"],
               s=13, color="#0072B2", marker="^", linewidths=0)
    ax.set_xlabel("Rigid free H$_2$ inventory (service hours)")
    ax.set_ylabel("Free H$_2$ inventory reduction (%)")
    ax.set_xlim(40, 540)
    ax.set_ylim(0, 46)
    handles = [Line2D([], [], ls="", marker="o", color="#8C8C8C", ms=3.6, label="Reference count unchanged"),
               Line2D([], [], ls="", marker="^", color="#0072B2", ms=3.6, label="Lower reference count")]
    ax.legend(handles=handles, frameon=False, loc="upper right", handletextpad=0.4, borderaxespad=0.3)

    ax = axes[1]
    data = by_site.sort_values(["all_replay_module_count_rigid", "site_id"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(data))
    ax.hlines(y, data.all_replay_module_count_flex, data.all_replay_module_count_rigid, color="#BFBFBF", lw=1.4, zorder=1)
    # Hollow outer rings keep both outcomes visible when counts coincide.
    ax.plot(data.all_replay_module_count_rigid, y, ls="", marker="o", ms=5.4, color=COLORS["rigid"],
            markerfacecolor="white", markeredgecolor=COLORS["rigid"], markeredgewidth=0.7, zorder=3)
    ax.plot(data.all_replay_module_count_flex, y, ls="", marker="s", ms=3.1, color=COLORS["flex_ml0p6_b24"],
            markeredgecolor="white", markeredgewidth=0.35, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([SITE_LABEL[s] for s in data.site_id])
    ax.set_ylim(-0.75, len(data) - 0.25)
    ax.set_xlim(0.4, 8.6)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Reference units covering all nine annual replays")
    handles = [Line2D([], [], ls="", marker="o", color=COLORS["rigid"], ms=5.4,
                      markerfacecolor="white", markeredgewidth=0.7, label="Rigid service"),
               Line2D([], [], ls="", marker="s", color=COLORS["flex_ml0p6_b24"], ms=3.1, label="60% floor, 24 h buffer")]
    ax.legend(handles=handles, frameon=False, loc="lower right", handletextpad=0.4, borderaxespad=0.3)
    panel_letter(axes[0], "a", dx=0.015, dy=0.965)
    panel_letter(axes[1], "b", dx=0.015, dy=0.965)
    save(fig, "figure2_inventory_and_design")


def figure3(bounds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(mm(WIDTH_NARROW_MM), mm(54)), constrained_layout=True)
    order = ["rigid", "flex_ml0p6_b24", "flex_ml0p4_b24"]
    certified, equality = [], []
    for case in order:
        sub = bounds[bounds.process_case == case]
        certified.append(int(sub.certified_deliverability_above_volume_bound.sum()))
        equality.append(int((sub.combined_necessary_module_lower_bound == sub.optimized_reference_modules).sum()))
    y = np.arange(len(order))
    h = 0.34
    ax.barh(y + h / 2, certified, height=h, color="#0072B2", label="Rate bound above the volume bound")
    ax.barh(y - h / 2, equality, height=h, color="#56B4E9", label="Necessary bound equals the optimized count")
    for yi, (c, e) in enumerate(zip(certified, equality)):
        ax.text(c + 2.5, yi + h / 2, f"{c}", va="center", ha="left")
        ax.text(e + 2.5, yi - h / 2, f"{e}", va="center", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Location\u2013year configurations (of 135)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, handlelength=1.4)
    save(fig, "figure3_independent_rate_certificates")


def figure4(rate: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(mm(WIDTH_MM), mm(62)), sharey=True, constrained_layout=True)
    offsets = {case: 0.0 for case in CASES}
    for ax, sid in zip(axes, ANCHORS):
        for case in CASES:
            sub = rate[(rate.site_id == sid) & (rate.process_case == case)].sort_values("weather_year")
            x = [YEARS.index(y) + offsets[case] for y in sub.weather_year]
            y = sub.minimum_withdrawal_kgph_per_normalized_tph.to_numpy(dtype=float)
            ax.plot(x, y, color=COLORS[case], marker=MARKERS[case], lw=0.9, ms=3.6,
                    markerfacecolor=COLORS[case], markeredgecolor="white", markeredgewidth=0.5)
        year_axis(ax)
        ax.set_title(ANCHOR_LABEL[sid], loc="left")
    axes[0].set_ylabel("Minimum withdrawal capacity\n(kg H$_2$ h$^{-1}$ per t NH$_3$ h$^{-1}$)")
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], lw=0.9, ms=3.6,
                      markerfacecolor=COLORS[c], markeredgecolor="white", markeredgewidth=0.5,
                      label=LABELS[c]) for c in CASES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handlelength=2.2)
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter, dx=0.015, dy=0.985)
    save(fig, "figure4_common_volume_deliverability")


def main() -> int:
    global MAIN, FRONTIER, RATE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.results_root:
        base = args.results_root.resolve()
        MAIN, FRONTIER, RATE = base / "primary", base / "frontier", base / "common_volume"
    if args.output_dir:
        OUT = args.output_dir.resolve()
    AUDIT.clear()
    style()
    frontier = pd.read_csv(FRONTIER / "continuous_module_frontier.csv")
    paired = pd.read_csv(MAIN / "paired_site_weather_summary_flex_ml0p6_b24.csv")
    by_site = pd.read_csv(MAIN / "summary_by_site_flex_ml0p6_b24.csv")
    bounds = pd.read_csv(MAIN / "capacity_deliverability_bounds.csv")
    rate = pd.read_csv(RATE / "fixed_volume_deliverability_counterfactual.csv")

    assert len(frontier) == 27 and frontier.status.eq("optimal").all()
    assert len(paired) == 135
    assert len(by_site) == 15
    assert len(bounds) == 405
    assert len(rate) == 27 and rate.status.eq("optimal").all()

    figure1(frontier)
    figure2(paired, by_site)
    figure3(bounds)
    figure4(rate)

    problems = [p for entry in AUDIT for p in entry["problems"]]
    report = {
        "figures": AUDIT,
        "problems": problems,
        "dpi_png": 1200,
        "scienceplots_commit": STYLE_COMMIT,
        "scienceplots_presets": ["science", "nature", "no-latex"],
        "vector_formats": ["pdf", "svg", "eps"],
        "width_mm": WIDTH_MM,
        "source_tables": {
            "figure1": str(FRONTIER / "continuous_module_frontier.csv"),
            "figure2": [
                str(MAIN / "paired_site_weather_summary_flex_ml0p6_b24.csv"),
                str(MAIN / "summary_by_site_flex_ml0p6_b24.csv"),
            ],
            "figure3": str(MAIN / "capacity_deliverability_bounds.csv"),
            "figure4": str(RATE / "fixed_volume_deliverability_counterfactual.csv"),
        },
    }
    inputs = [FRONTIER / "continuous_module_frontier.csv",
              MAIN / "paired_site_weather_summary_flex_ml0p6_b24.csv",
              MAIN / "summary_by_site_flex_ml0p6_b24.csv",
              MAIN / "capacity_deliverability_bounds.csv",
              RATE / "fixed_volume_deliverability_counterfactual.csv"]
    report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    # Portable public provenance: no machine-specific absolute path is needed.
    report["source_tables"] = {
        "figure1": "results/frontier/continuous_module_frontier.csv",
        "figure2": ["results/primary/paired_site_weather_summary_flex_ml0p6_b24.csv",
                    "results/primary/summary_by_site_flex_ml0p6_b24.csv"],
        "figure3": "results/primary/capacity_deliverability_bounds.csv",
        "figure4": "results/common_volume/fixed_volume_deliverability_counterfactual.csv",
    }
    (OUT / "figure_layout_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"layout_problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
