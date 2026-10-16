"""Dependency-free SVG grouped-bar chart used by the Module A studies (#37).

Colours are the fixed categorical order of the dataviz reference palette (light surface), assigned
by series position and never cycled; every chart has a legend and the studies also commit the
numbers as markdown tables, so identity never rests on colour alone.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _nice_step(span: float, n: int) -> float:
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def grouped_bars_svg(
    title: str,
    groups: list[str],
    series: dict[str, list[float]],
    ylabel: str,
    errors: dict[str, list[float]] | None = None,
    path: str | Path | None = None,
) -> str:
    """``series[name][i]`` is the value of ``name`` in ``groups[i]``; optional symmetric ``errors``."""
    names = list(series)
    if len(names) > len(SERIES_COLORS):
        raise ValueError("more series than categorical colours; facet instead")
    left, top, right, bottom = 64, 52, 24, 56
    gw = max(120, 18 * len(names) + 30, -(-(560 - left - right) // len(groups)))
    w = left + right + gw * len(groups)
    col_w = max(len(n) for n in names) * 6.4 + 30
    cols = max(1, int((gw * len(groups)) // col_w))
    legend_rows = -(-len(names) // cols)
    h = 292 + 18 * legend_rows
    plot_h = 300 - top - bottom
    hi = max([v + (errors[n][i] if errors else 0.0) for n in names for i, v in enumerate(series[n])] + [1e-9])
    lo = min([0.0] + [v for n in names for v in series[n]])
    step = _nice_step((hi - lo) or 1.0, 5)
    hi, lo = math.ceil(hi / step - 1e-9) * step, math.floor(lo / step + 1e-9) * step
    span = (hi - lo) or step
    ticks = round(span / step)

    def y(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / span)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
           f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
           f'<text x="{left}" y="24" font-size="14" font-weight="600" fill="{INK}">{escape(title)}</text>']
    for k in range(ticks + 1):
        v = lo + span * k / ticks
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{v:.3g}</text>')
    out.append(f'<text x="14" y="{top + plot_h / 2:.0f}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {top + plot_h / 2:.0f})" text-anchor="middle">{escape(ylabel)}</text>')
    bw = (gw - 30) / len(names)
    for gi, g in enumerate(groups):
        x0 = left + gi * gw + 15
        for si, n in enumerate(names):
            v = series[n][gi]
            x = x0 + si * bw
            y0, y1 = y(0.0), y(v)
            out.append(f'<rect x="{x + 1:.1f}" y="{min(y0, y1):.1f}" width="{max(bw - 2, 1):.1f}" height="{abs(y0 - y1):.1f}" rx="2" fill="{SERIES_COLORS[si]}"><title>{escape(n)} / {escape(g)}: {v:.3g}</title></rect>')
            if errors:
                e = errors[n][gi]
                if e:
                    cx = x + bw / 2
                    out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(v - e):.1f}" y2="{y(v + e):.1f}" stroke="{INK}" stroke-width="1"/>')
        out.append(f'<text x="{x0 + bw * len(names) / 2:.1f}" y="{300 - bottom + 18}" font-size="11" text-anchor="middle" fill="{INK}">{escape(g)}</text>')
    out.append(f'<line x1="{left}" x2="{w - right}" y1="{y(0.0):.1f}" y2="{y(0.0):.1f}" stroke="{MUTED}" stroke-width="1"/>')
    for si, n in enumerate(names):
        lx = left + (si % cols) * col_w
        ly = 300 - 14 + 18 * (si // cols)
        out.append(f'<rect x="{lx:.0f}" y="{ly - 9}" width="10" height="10" rx="2" fill="{SERIES_COLORS[si]}"/>')
        out.append(f'<text x="{lx + 15:.0f}" y="{ly}" font-size="11" fill="{INK}">{escape(n)}</text>')
    out.append("</svg>")
    svg = "\n".join(out)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(svg, encoding="utf-8")
    return svg
