#!/usr/bin/env python3
"""将 analyze.py 的汇总 JSON 转成四张独立 SVG。"""

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


WIDTH = 1180
MODELS = (("1.7b", "1.7B", 1.7), ("4b", "4B", 4), ("8b", "8B", 8))
CONDITIONS = (
    ("edit", "方向编辑"),
    ("random", "随机方向"),
    ("concise", "简洁提示"),
    ("reseed", "换种子"),
)
COLORS = {
    "background": ("#ffffff", "#141922"),
    "panel": ("#f6f8fb", "#1d2633"),
    "ink": ("#182635", "#f2f5fa"),
    "muted": ("#46586a", "#b7c5d5"),
    "rule": ("#cdd7e2", "#435266"),
    "axis": ("#65768a", "#91a2b8"),
    "blue": ("#006ca6", "#62c1f5"),
    "orange": ("#a9460b", "#ffac69"),
    "green": ("#007561", "#68d6b8"),
    "purple": ("#81428d", "#dc9be7"),
    "gray": ("#57697c", "#adbbcc"),
}


def stylesheet():
    rules = [
        'text { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", '
        '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif; '
        'font-variant-numeric: tabular-nums; }',
        '.label-halo { paint-order: stroke; stroke-width: 3; stroke-linejoin: round; }',
    ]

    def palette(theme):
        return "\n".join(
            f".fill-{key} {{ fill: {colors[theme]}; }} "
            f".stroke-{key} {{ stroke: {colors[theme]}; }}"
            for key, colors in COLORS.items()
        )

    rules.append(palette(0))
    rules.append("@media (prefers-color-scheme: dark) {\n" + palette(1) + "\n}")
    return "\n".join(rules)


class Figure:
    def __init__(self, height, title, subtitle, synthetic):
        self.root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "xml:lang": "zh-CN",
            "width": str(WIDTH), "height": str(height),
            "viewBox": f"0 0 {WIDTH} {height}",
            "style": "max-width: 100%; height: auto;",
            "role": "img", "aria-labelledby": "title description",
        })
        self.add("title", id="title").text = (
            f"{title}（合成示例）" if synthetic else title
        )
        self.add("desc", id="description").text = subtitle
        self.add("style").text = stylesheet()
        self.rect(0, 0, WIDTH, height, "background")
        self.text(36, 49, title, size=30, bold=True)
        self.text(36, 79, subtitle, size=15, color="muted")
        if synthetic:
            self.rect(996, 25, 148, 34, "purple", radius=8)
            self.text(1070, 47, "合成示例 · 假数据", size=14, color="background",
                      anchor="middle", bold=True)

    def add(self, tag, **attributes):
        return ET.SubElement(self.root, tag, {
            key.removesuffix("_").replace("_", "-"): str(value)
            for key, value in attributes.items()
        })

    def text(self, x, y, label, *, size=14, color="ink", anchor="start", bold=False, halo=False):
        classes = f"fill-{color}" + (" stroke-background label-halo" if halo else "")
        self.add("text", x=x, y=y, class_=classes, font_size=size,
                 text_anchor=anchor, font_weight=650 if bold else 400).text = label

    def rect(self, x, y, width, height, color, *, radius=0):
        self.add("rect", x=x, y=y, width=width, height=height,
                 rx=radius, class_=f"fill-{color}")

    def line(self, x1, y1, x2, y2, color, *, width=1, dash=None):
        style = {"stroke_dasharray": dash} if dash else {}
        self.add("line", x1=x1, y1=y1, x2=x2, y2=y2,
                 class_=f"stroke-{color}", stroke_width=width, **style)

    def marker(self, x, y, shape, color):
        if shape == "circle":
            self.add("circle", cx=x, cy=y, r=4.5, class_=f"fill-{color}")
        elif shape == "square":
            self.add("rect", x=x-6, y=y-6, width=12, height=12,
                     class_=f"fill-background stroke-{color}", stroke_width=2)
        else:
            points = f"{x},{y-6} {x+6},{y} {x},{y+6} {x-6},{y}"
            self.add("polygon", points=points, class_=f"fill-background stroke-{color}",
                     stroke_width=2.3)

    def legend(self, x, label, color, symbol="bar"):
        y = 111
        if symbol == "bar":
            self.rect(x, y-10, 22, 12, color, radius=2)
        elif symbol == "diamond":
            self.marker(x+11, y-4, symbol, color)
        else:
            dash = "6 4" if symbol in ("dash", "square") else None
            self.line(x, y-4, x+22, y-4, color, width=2, dash=dash)
            if symbol != "dash":
                self.marker(x+11, y-4, symbol, color)
        self.text(x+31, y, label)

    def save(self, path):
        ET.indent(self.root, space="  ")
        ET.ElementTree(self.root).write(path, encoding="utf-8", xml_declaration=True)


def finite(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def number_label(value):
    return f"{100 * value:.2f}".rstrip("0").rstrip(".").replace("-", "−")


def signed_label(value):
    return ("+" if value > 0 else "") + number_label(value)


def baseline(model):
    for condition in (model.get("conditions") or {}).values():
        if not condition:
            continue
        accuracy = finite(condition.get("base_acc"))
        truncation = finite(condition.get("base_trunc"))
        if accuracy is not None and truncation is not None:
            return accuracy, truncation
    return None, None


def gain_ticks(values):
    """选择包含零、能容纳全部值的百分点刻度。"""
    low, high = min([0, *values]) * 100, max([0, *values]) * 100
    span = max(high - low, 0.1)
    step = next(step for step in (0.05, 0.1, 0.2, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100)
                if step * 6 >= span)
    start = math.floor(round(low / step, 10))
    stop = math.ceil(round(high / step, 10))
    if start == stop:
        return (-0.01, 0, 0.01)
    return tuple(i * step / 100 for i in range(start, stop + 1))


def y_axis(fig, left, right, top, bottom, ticks, label):
    low, high = ticks[0], ticks[-1]

    def y(value):
        return bottom - (value-low) / (high-low) * (bottom-top)

    for tick in ticks:
        grid_y = y(tick)
        fig.line(left, grid_y, right, grid_y, "axis" if tick == 0 else "rule")
        fig.text(left-7, grid_y+4, label(tick), size=11, color="muted", anchor="end")
    return y


def panel(fig, index, model_label):
    x = 36 + index * 376
    fig.rect(x, 135, 356, 351, "panel", radius=12)
    fig.text(x+16, 165, f"Qwen3-{model_label}", size=19, bold=True)
    return x+49, x+342


def missing(fig, x, y, label="缺失"):
    fig.text(x, y, label, size=12, color="muted", anchor="middle")


def rates(models, synthetic):
    fig = Figure(536, "各条件的准确率与截断率",
                 "每个面板对应一个模型尺寸，两个指标均以全部留出题为分母。", synthetic)
    fig.legend(36, "准确率", "blue")
    fig.legend(144, "截断率", "orange")
    for index, (label, _, model) in enumerate(models):
        left, right = panel(fig, index, label)
        bottom = 418
        y = y_axis(fig, left, right, 207, bottom, (0, .25, .5, .75, 1),
                   lambda tick: number_label(tick) + "%")
        conditions = model.get("conditions") or {}
        data = [("基线", baseline(model))]
        for key, name in CONDITIONS:
            stats = conditions.get(key) or {}
            data.append((name, (finite(stats.get("edit_acc")), finite(stats.get("edit_trunc")))))
        for column, (name, (acc, trunc)) in enumerate(data):
            x = left + (column+.5) * (right-left) / len(data)
            fig.text(x, 452, name, size=12, anchor="middle")
            if acc is None and trunc is None:
                missing(fig, x, 404)
                continue
            for value, center, color in ((acc, x-13, "blue"), (trunc, x+13, "orange")):
                if value is None:
                    missing(fig, center, 404, "—")
                    continue
                bar_top = y(value)
                fig.rect(center-9, bar_top, 18, bottom-bar_top, color)
                stagger = color == "orange" and acc is not None and abs(y(acc)-bar_top) < 13
                fig.text(center, bar_top-(18 if stagger else 6), number_label(value),
                         size=10, color=color, anchor="middle", bold=True)
    fig.text(36, 517, "柱顶数值单位为 %。达到 token 上限的答案计错。", size=13, color="muted")
    return fig


def decomposition(models, synthetic):
    fig = Figure(536, "准确率变化 Δacc 的两项分解",
                 "同号分量堆叠，异号分量分列零轴两侧。菱形标示两项相加后的净变化。", synthetic)
    fig.legend(36, "截断率变化贡献", "blue")
    fig.legend(209, "答完后准确率变化贡献", "orange")
    fig.legend(430, "净 Δacc", "ink", "diamond")
    fig.legend(542, "换种子噪声参考（reseed Δacc）", "purple", "dash")
    values = []
    data = []
    for _, _, model in models:
        conditions = model.get("conditions") or {}
        row = []
        for key, _ in CONDITIONS:
            stats = conditions.get(key) or {}
            parts = (finite(stats.get("from_less_truncation")),
                     finite(stats.get("from_finished_accuracy")))
            delta = finite(stats.get("delta"))
            if delta is not None:
                values.append(delta)
            if None not in parts:
                values += [sum(min(0, part) for part in parts), sum(max(0, part) for part in parts)]
            row.append((parts, delta))
        data.append(row)
    ticks = gain_ticks(values)
    for index, ((label, _, _), row) in enumerate(zip(models, data)):
        left, right = panel(fig, index, label)
        noise = row[-1][1]
        if noise is not None:
            fig.text(right, 165, f"换种子 {signed_label(noise)}", size=12,
                     color="purple", anchor="end", bold=True)
        y = y_axis(fig, left, right, 207, 420, ticks, signed_label)
        markers = []
        for column, ((_, name), (parts, delta)) in enumerate(zip(CONDITIONS, row)):
            x = left + (column+.5) * (right-left) / len(row)
            fig.text(x, 456, name, size=12, anchor="middle")
            positive = negative = 0
            if None not in parts:
                for part, color in zip(parts, ("blue", "orange")):
                    if part >= 0:
                        end = positive + part
                        fig.rect(x-16, y(end), 32, y(positive)-y(end), color)
                        positive = end
                    else:
                        end = negative + part
                        fig.rect(x-16, y(negative), 32, y(end)-y(negative), color)
                        negative = end
            elif delta is None:
                missing(fig, x, y(0)-12)
            if delta is not None:
                label_y = y(max(positive, delta))-11 if delta >= 0 else y(min(negative, delta))+17
                markers.append((x, delta, label_y))
        if noise is not None:
            fig.line(left, y(noise), right, y(noise), "purple", width=1.8, dash="6 4")
        for x, delta, label_y in markers:
            fig.marker(x, y(delta), "diamond", "ink")
            fig.text(x, label_y, signed_label(delta), size=11, anchor="middle", bold=True)
    fig.text(36, 517, "纵轴单位为百分点。虚线为单次换种子对照。分解项缺失时保留可用的净变化。",
             size=13, color="muted")
    return fig


def scaling(models, synthetic):
    fig = Figure(545, "方向编辑与 2 倍预算的规模曲线",
                 "两个系列均相对同尺寸的留出题基线计算准确率变化。", synthetic)
    fig.legend(36, "方向编辑", "blue", "circle")
    fig.legend(177, "2 倍预算（续写）", "orange", "square")
    gains = []
    for _, _, model in models:
        edit = (model.get("conditions") or {}).get("edit") or {}
        cont = model.get("continuation") or {}
        acc, _ = baseline(model)
        extended = finite(cont.get("base_acc_with_2x_budget"))
        gains.append((finite(edit.get("delta")),
                      extended-acc if extended is not None and acc is not None else None))
    ticks = gain_ticks([value for pair in gains for value in pair if value is not None])
    left, right = 99, 1118
    top, bottom = 170, 432
    y = y_axis(fig, left, right, top, bottom, ticks, signed_label)
    fig.text(left, 148, "准确率变化（百分点）", size=13, color="muted")
    positions = []
    for (label, size, _), pair in zip(models, gains):
        x = left + (size-1) * (right-left) / 8
        positions.append(x)
        fig.line(x, top, x, bottom, "rule", dash="3 5")
        fig.text(x, bottom+24, label, size=15, anchor="middle", bold=True)
        if pair == (None, None):
            missing(fig, x, top+20, "暂无数据")
    for series, (color, dash) in enumerate((("blue", None), ("orange", "6 4"))):
        previous = None
        for x, pair in zip(positions, gains):
            value = pair[series]
            if value is None:
                previous = None
                continue
            point = (x, y(value))
            if previous is not None:
                fig.line(*previous, *point, color, width=2.5, dash=dash)
            previous = point
    for x, (edit, budget) in zip(positions, gains):
        if budget is not None:
            fig.marker(x, y(budget), "square", "orange")
            fig.text(x+12, y(budget)-9, signed_label(budget), size=13, color="orange",
                     bold=True, halo=True)
        if edit is not None:
            fig.marker(x, y(edit), "circle", "blue")
            fig.text(x-12, y(edit)-9, signed_label(edit), size=13, color="blue",
                     anchor="end", bold=True, halo=True)
    fig.text((left+right)/2, 485, "参数规模（B 表示十亿）", size=14, anchor="middle")
    fig.text(36, 524, "2 倍预算只在基线截断前缀上再续写一次。缺失的点留空，连线在该处中断。",
             size=13, color="muted")
    return fig


def continuation(models, synthetic):
    fig = Figure(488, "截断答案在续写后的去向",
                 "每行以该模型在原 token 上限时被截断的答案数为分母。", synthetic)
    categories = (("续写后答对", "green"), ("续写后答错", "orange"), ("仍未答完", "gray"))
    for x, (label, color) in zip((36, 185, 334), categories):
        fig.legend(x, label, color)
    left, right = 222, 1144
    for i in range(5):
        x = left + i * (right-left) / 4
        fig.text(x, 148, f"{i*25}%", size=12, anchor="middle", color="muted")
        fig.line(x, 158, x, 422, "rule", dash="3 4")
    for index, (label, _, model) in enumerate(models):
        y = 181 + 86 * index
        fig.text(36, y+22, f"Qwen3-{label}", size=19, bold=True)
        cont = model.get("continuation") or {}
        n, finished, correct = (cont.get(key) for key in
                                ("n_truncated", "finished_with_2x", "correct_with_2x"))
        if not all(type(count) is int for count in (n, finished, correct)) or not 0 <= correct <= finished <= n:
            missing(fig, (left+right)/2, y+21, "续写数据缺失或计数无效")
            continue
        fig.text(36, y+43, f"原截断 n = {n}", size=13, color="muted")
        if n == 0:
            missing(fig, (left+right)/2, y+21, "原基线无截断答案")
            continue
        x = left
        for i, (count, (name, color)) in enumerate(zip((correct, finished-correct, n-finished), categories)):
            width = (right-left) * count / n
            if width > 0:
                fig.rect(x, y, width, 30, color)
                if x > left:
                    fig.line(x, y, x, y+30, "background", width=1.5)
            x += width
            fig.text(left+i*300, y+52, f"{name} {number_label(count/n)}%（{count}）",
                     size=14, color=color)
    fig.text(36, 465, "答错数为续写后已答完数减答对数。括号内为答案数。", size=13, color="muted")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="analyze.py 导出的结果 JSON")
    parser.add_argument("--out", type=Path, required=True, help="SVG 输出目录")
    args = parser.parse_args()
    with args.results.open(encoding="utf-8") as source:
        results = json.load(source)
    if not isinstance(results, dict):
        parser.error("结果 JSON 的顶层必须是对象")
    models = [(label, size, results.get(key) or {}) for key, label, size in MODELS]
    synthetic = results.get("synthetic") is True
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, render in (
        ("01_rates.svg", rates),
        ("02_decomposition.svg", decomposition),
        ("03_scaling.svg", scaling),
        ("04_continuation.svg", continuation),
    ):
        target = args.out / filename
        render(models, synthetic).save(target)
        print(target)


if __name__ == "__main__":
    main()
