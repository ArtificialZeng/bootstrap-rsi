#!/usr/bin/env python3
"""把 out/q35_caps.json 的预算曲线导出为三个独立 SVG。"""

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


EXP = Path(__file__).resolve().parents[2]
WIDTH = 800
LEFT, RIGHT, TOP, BOTTOM = 86, 752, 146, 365
FOOTNOTE = (
    "Answers truncated at 4096 were continued to the larger cap. "
    "Intermediate caps are offline re-cuts."
)
COLORS = {
    "background": "#ffffff",
    "panel": "#f6f8fb",
    "ink": "#182635",
    "muted": "#46586a",
    "rule": "#cdd7e2",
    "axis": "#65768a",
    "blue": "#006ca6",
    "orange": "#a9460b",
    "green": "#007561",
    "purple": "#81428d",
    "gray": "#57697c",
}
ACCURACY_SERIES = (
    (("base", "reseed"), "Base (2 seeds)", "gray", None),
    (("s8", "s8@2000"), "S8 edit (2 seeds)", "blue", None),
    (("s8rand",), "Random sign", "purple", "3 5"),
    (("concise",), "Concise prompt", "green", "9 4"),
    (("s8concise",), "S8 + concise", "orange", "9 4 3 4"),
)
DELTA_SERIES = (
    ("s8 (2 seeds) - base (2 seeds)", "blue", None, -9),
    ("concise - base (2 seeds)", "green", "9 4", 0),
    ("s8rand - base", "purple", "3 5", 9),
)
ARM_LABELS = {
    "base": ("Base", "gray"),
    "reseed": ("Base (reseed)", "gray"),
    "s8": ("S8 edit", "blue"),
    "s8@2000": ("S8 edit (seed 2000)", "blue"),
    "s8rand": ("Random sign", "purple"),
    "concise": ("Concise prompt", "green"),
    "s8concise": ("S8 + concise", "orange"),
}


class Figure:
    def __init__(self, height, title):
        self.root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg", "xml:lang": "en",
            "width": str(WIDTH), "height": str(height),
            "viewBox": f"0 0 {WIDTH} {height}", "role": "img",
            "aria-labelledby": "title method", "style": "max-width:100%;height:auto",
        })
        self.add("title", id="title").text = title
        self.add("style").text = (
            "text {font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; "
            "font-variant-numeric: tabular-nums;}"
        )
        self.rect(0, 0, WIDTH, height, "background")
        self.text(36, 44, title, size=27, bold=True)
        self.text(36, height-20, FOOTNOTE, size=13, color="muted", id="method")

    def add(self, tag, **attrs):
        return ET.SubElement(self.root, tag, {
            key.replace("_", "-"): str(value) for key, value in attrs.items()
        })

    def text(self, x, y, content, *, size=14, color="ink", anchor="start", bold=False, **attrs):
        self.add("text", x=x, y=y, font_size=size, fill=COLORS[color], text_anchor=anchor,
                 font_weight=650 if bold else 400, **attrs).text = content

    def rect(self, x, y, width, height, color):
        self.add("rect", x=x, y=y, width=width, height=height, fill=COLORS[color])

    def line(self, x1, y1, x2, y2, color, *, width=1, dash=None, opacity=1):
        style = {"stroke_dasharray": dash} if dash else {}
        self.add("line", x1=x1, y1=y1, x2=x2, y2=y2,
                 stroke=COLORS[color], stroke_width=width, stroke_opacity=opacity, **style)

    def marker(self, x, y, color, *, hollow=False):
        self.add("circle", cx=x, cy=y, r=4, stroke=COLORS[color], stroke_width=2,
                 fill=COLORS["background" if hollow else color])

    def legend(self, x, label, color, dash):
        self.line(x, 75, x+23, 75, color, width=2.4, dash=dash)
        self.marker(x+11.5, 75, color)
        self.text(x+32, 80, label, size=14)

    def save(self, path):
        ET.indent(self.root, space="  ")
        ET.ElementTree(self.root).write(path, encoding="utf-8", xml_declaration=True)


def mapping(value):
    return value if isinstance(value, dict) else {}


def finite(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def load_caps(path):
    with path.open(encoding="utf-8") as source:
        data = mapping(json.load(source))
    return dict(sorted((int(cap), row) for cap, row in data.items()
                       if cap.isascii() and cap.isdecimal() and int(cap) > 0
                       and isinstance(row, dict)))


def nice_ticks(low, high):
    if low == high:
        return (low, low+1)
    target = (high-low) / 6
    magnitude = 10 ** math.floor(math.log10(target))
    step = next(magnitude * factor for factor in (1, 2, 2.5, 5, 10)
                if magnitude * factor >= target)
    start = math.floor(round(low / step, 10))
    end = math.ceil(round(high / step, 10))
    return tuple(i * step for i in range(start, end+1))


def as_percent(value):
    return f"{value:.8g}".replace("-", "−") + "%"


def as_pp(value):
    return f"{value:+.8g}".replace("-", "−") if value else "0"


def cap_axes(fig, caps, ticks, ylabel, format_tick):
    fig.rect(LEFT, TOP, RIGHT-LEFT, BOTTOM-TOP, "panel")
    low, high = ticks[0], ticks[-1]

    def y(value):
        return BOTTOM-(value-low) * (BOTTOM-TOP) / (high-low)

    if len(caps) > 1:
        first, last = min(caps), max(caps)
        positions = {cap: LEFT+15+(cap-first) * (RIGHT-LEFT-30) / (last-first) for cap in caps}
    else:
        positions = {cap: (LEFT+RIGHT)/2 for cap in caps}
    for cap, x in positions.items():
        fig.line(x, TOP, x, BOTTOM, "rule", dash="3 5")
        fig.text(x, BOTTOM+23, f"{cap:,}", color="muted", anchor="middle")
    for tick in ticks:
        line_y = y(tick)
        fig.line(LEFT, line_y, RIGHT, line_y, "axis" if tick == 0 else "rule",
                 width=1.5 if tick == 0 else 1)
        fig.text(LEFT-10, line_y+5, format_tick(tick), color="muted", anchor="end")
    fig.text(LEFT, TOP-10, ylabel, color="muted")
    fig.text((LEFT+RIGHT)/2, BOTTOM+53, "Output token cap", anchor="middle")
    return positions, y


def accuracy_values(row, keys):
    arms = mapping(row.get("arms"))
    values = [finite(mapping(arms.get(key)).get("acc")) for key in keys]
    return [100 * value for value in values if value is not None and 0 <= value <= 1]


def seed_band(fig, points, y, color):
    if len(points) == 1:
        (x, low, high), = points
        fig.line(x, y(low), x, y(high), color, width=8, opacity=0.18)
    elif points:
        upper = [(x, y(high)) for x, _, high in points]
        lower = [(x, y(low)) for x, low, _ in reversed(points)]
        fig.add("polygon", points=" ".join(f"{x},{py}" for x, py in upper+lower),
                fill=COLORS[color], fill_opacity=0.18)


def accuracy_chart(caps):
    fig = Figure(474, "Accuracy by token cap")
    fig.text(36, 106, "Shading spans both seeds. Open markers show one available seed.",
             color="muted")
    series = [[accuracy_values(row, keys) for row in caps.values()]
              for keys, _, _, _ in ACCURACY_SERIES]
    values = [value for points in series for available in points for value in available]
    ticks = nice_ticks(0, max(values, default=100))
    positions, y = cap_axes(fig, caps, ticks, "Accuracy", as_percent)
    if not values:
        fig.text((LEFT+RIGHT)/2, (TOP+BOTTOM)/2, "No accuracy data available",
                 color="muted", anchor="middle")

    for (_, _, color, _), points in zip(ACCURACY_SERIES, series):
        run = []
        for x, available in zip(positions.values(), points):
            if len(available) == 2:
                run.append((x, min(available), max(available)))
            else:
                seed_band(fig, run, y, color)
                run = []
        seed_band(fig, run, y, color)

    for legend_x, (keys, label, color, dash), points in zip(
            (36, 184, 352, 493, 642), ACCURACY_SERIES, series):
        if any(points):
            fig.legend(legend_x, label, color, dash)
        previous = None
        for x, available in zip(positions.values(), points):
            if not available:
                previous = None
                continue
            py = y(sum(available)/len(available))
            single_seed = len(available) < len(keys)
            if previous is not None and not single_seed:
                fig.line(*previous, x, py, color, width=2.4, dash=dash)
            fig.marker(x, py, color, hollow=single_seed)
            previous = None if single_seed else (x, py)
    return fig


def delta_point(row, pair):
    stats = mapping(mapping(row.get("pairs")).get(pair))
    delta = finite(stats.get("delta"))
    if delta is None:
        return None
    interval = stats.get("ci")
    if isinstance(interval, list) and len(interval) == 2:
        low, high = map(finite, interval)
        if low is not None and high is not None and low <= high:
            return 100*delta, 100*low, 100*high
    return 100*delta, None, None


def delta_chart(caps):
    fig = Figure(474, "Accuracy change by token cap")
    fig.text(36, 106, "Error bars show 95% intervals. Markers are offset horizontally for readability.",
             color="muted")
    series = [[delta_point(row, pair) for row in caps.values()] for pair, _, _, _ in DELTA_SERIES]
    values = [value for points in series for point in points if point is not None
              for value in point if value is not None]
    low, high = (min(0, *values), max(0, *values)) if values else (-1, 1)
    if low == high:
        low, high = -1, 1
    positions, y = cap_axes(fig, caps, nice_ticks(low, high), "Change (percentage points)", as_pp)
    if not values:
        fig.text((LEFT+RIGHT)/2, (TOP+BOTTOM)/2-12, "No paired comparisons available",
                 color="muted", anchor="middle")

    for legend_x, (pair, color, dash, offset), points in zip((36, 305, 556), DELTA_SERIES, series):
        if any(points):
            fig.legend(legend_x, pair, color, dash)
        previous = None
        available = []
        for x, point in zip(positions.values(), points):
            if point is None:
                previous = None
                continue
            delta, low, high = point
            x += offset
            py = y(delta)
            if previous is not None:
                fig.line(*previous, x, py, color, width=2.4, dash=dash)
            previous = (x, py)
            available.append((x, py, low, high))
        for x, py, low, high in available:
            if low is not None:
                fig.line(x, y(low), x, y(high), color, width=1.6)
                for bound in (low, high):
                    fig.line(x-4, y(bound), x+4, y(bound), color, width=1.6)
            fig.marker(x, py, color)
    return fig


def tokens_per_correct(stats):
    stats = mapping(stats)
    if finite(stats.get("acc")) == 0:
        return None
    value = finite(stats.get("tok_per_correct"))
    return value if value is not None and value >= 0 else None


def tokens_chart(caps):
    arms_at_cap = {cap: mapping(row.get("arms")) for cap, row in caps.items()}
    usable = {cap for cap, arms in arms_at_cap.items()
              if any(tokens_per_correct(stats) is not None for stats in arms.values())}
    cap = 8192 if 8192 in usable else max(usable, default=None)
    arms = arms_at_cap.get(cap, {})
    keys = [key for key in ARM_LABELS if key in arms] + sorted(arms.keys()-ARM_LABELS.keys())
    values = [tokens_per_correct(arms[key]) for key in keys]
    height = 193+36*max(1, len(keys))
    fig = Figure(height, "Output tokens per correct answer")
    if cap is None:
        fig.text(36, 90, "No token-to-correct metrics available", color="muted")
        return fig
    fig.text(36, 77, f"{cap:,}-token cap", color="muted")
    left, right, top, bottom = 213, 705, 106, 106+36*len(keys)
    ticks = nice_ticks(0, max(value for value in values if value is not None))

    def x(value):
        return left+value * (right-left) / ticks[-1]

    fig.rect(left, top, right-left, bottom-top, "panel")
    for tick in ticks:
        tick_label = f"{tick:,.0f}" if ticks[-1] >= 10 else f"{tick:.3g}"
        fig.line(x(tick), top, x(tick), bottom, "axis" if tick == 0 else "rule")
        fig.text(x(tick), bottom+23, tick_label, anchor="middle", color="muted")
    for index, (key, value) in enumerate(zip(keys, values)):
        py = top+36*index+18
        label, color = ARM_LABELS.get(key, (key, "gray"))
        fig.text(left-12, py+5, label, anchor="end")
        if value is None:
            note = "Undefined (zero correct)" if mapping(arms[key]).get("acc") == 0 else "Unavailable"
            fig.text(left+9, py+5, note, color="muted")
        else:
            fig.rect(left, py-10, x(value)-left, 20, color)
            fig.text(x(value)+8, py+5, f"{value:,.0f}", color=color, bold=True)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "handoff/codex/figs_caps",
                        help="SVG 输出目录")
    args = parser.parse_args()
    if args.out.resolve().is_relative_to((EXP / "out").resolve()):
        parser.error("SVG 输出目录不能位于实验 out/ 内")
    caps = load_caps(EXP / "out/q35_caps.json")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, render in (("acc_vs_cap.svg", accuracy_chart), ("delta_vs_cap.svg", delta_chart),
                         ("tokens_per_correct.svg", tokens_chart)):
        path = args.out / name
        render(caps).save(path)
        print(path)


if __name__ == "__main__":
    main()
