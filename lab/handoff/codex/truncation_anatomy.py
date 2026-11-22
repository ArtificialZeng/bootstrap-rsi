#!/usr/bin/env python3
"""只读分析本地截断轨迹。用 -I 运行，报告写到标准输出。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from math_verify import parse, verify
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
HF = Path.home() / ".cache/huggingface/hub"
DEFAULT_INPUTS = (
    ROOT / "out/practice_base.pkl",
    ROOT / "out/mech_1.7b_eval_base.part0.pkl",
)
TOKENIZERS = {
    "1.7b": "models--Qwen--Qwen3-1.7B",
    "4b": "models--Qwen--Qwen3-4B",
    "8b": "models--mlx-community--Qwen3-8B-8bit",
    "q35-4b": "models--mlx-community--Qwen3.5-4B-MLX-bf16",
}
CLASSES = ("循环", "反复验算", "仍在推进", "其他")
COMPRESS_CHARS = 6000
COMPRESS_RATIO = 0.15
REPEAT_TAIL = 1250
REPEAT_RECENT = 480
REPEAT_LENGTH = 60
CHECK_TAIL = 2400
CHECK_FOLLOW = 80
PHRASE_LIMIT = 240

BOX = re.compile(r"\\boxed\s*\{")
BOX_JOIN = re.compile(r"\s*(?:[,，]\s*(?:and\b\s*)?|and\b\s*|or\b\s*)\s*")
ANSWER = re.compile(
    r"\b(?:the\s+)?(?:(?:final|correct)\s+)?answer\s+"
    r"(?:is|should\s+be|would\s+be|must\s+be)\b[ \t]*"
    r"|\b(?:final\s+)?answer[ \t]*:[ \t]*",
    re.I,
)
ANSWER_BACKWARD = re.compile(
    r"\b(?:I\s+(?:think|believe)|(?:it\s+)?seems\s+like|(?:safely\s+)?assume\s+that)\s+"
    r"(?P<value>[^?!;\n]{1,120}?)\s+is\s+(?:the\s+)?(?:correct|final)\s+answer\b",
    re.I,
)
ANSWER_STOP = re.compile(
    r"\n|[?!;]|\.(?!\d)|\s*,?\s+\b(?:but|however|because|which|since)\b",
    re.I,
)
ANSWER_HEDGE = re.compile(r"^(?:(?:probably|perhaps|indeed|just|simply|exactly|only)\s+)+", re.I)
NON_VALUE = re.compile(
    r"^(?:supposed|expected|going|likely\s+to|to\s+be|a\b|an\b|the\b|correct\b|"
    r"wrong\b|different\b|same\b|what\b|not\b|possible\b|in\s+terms\b|unclear\b|"
    r"(?:that|this|it)\s*$)",
    re.I,
)
CONDITIONAL = re.compile(r"\b(?:if|whether|that\s+means)\s*$", re.I)
CHECK = re.compile(
    r"\b(?:double[- ]check|recheck|recalculate|recompute|rederive|"
    r"(?:check|verify|confirm|think|try|work|compute|calculate)\s+(?:it\s+)?"
    r"(?:again|once\s+more|once\s+again|another\s+way|from\s+scratch)|"
    r"(?:let\s+me|need\s+to|want\s+to)\s+(?:(?:just|also)\s+)?(?:check|verify|confirm)|"
    r"(?:just\s+to|to)\s+(?:be\s+sure|make\s+sure|verify|double[- ]check)|"
    r"another\s+(?:method|approach|way)|"
    r"alternatively\b[^\n.!?]{0,45}\b(?:try|check|compute|calculate|derive|use|solve|consider))\b",
    re.I,
)
SURFACE = re.compile(r"[a-z]+|\d+|[^\s]", re.I)
DIGIT_GROUP = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?!\d)")
UNICODE_MATH = str.maketrans({"−": "-", "∕": "/", "×": "*", "∞": r"\infty ", "∪": r"\cup "})


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    value: str
    source: str


@dataclass
class Trace:
    row: dict
    path: Path
    category: str
    reason: str
    first: Candidate | None
    first_correct: bool
    found_correct: Candidate | None
    correct_box: bool
    has_visible_box: bool
    visible_correct_box: bool
    closed: bool
    first_token: int | None
    token_denominator: int | None
    token_source: str
    zlib_ratio: float
    repeat_length: int
    excerpt: str


def matching_brace(text: str, start: int) -> int | None:
    depth = 0
    for match in re.finditer(r"\\[{}\\]|[{}]", text[start:]):
        brace = match.group()
        if brace == "{":
            depth += 1
        elif brace == "}":
            depth -= 1
            if depth == 0:
                return start + match.end()
    return None


def boxes(text: str, start: int, end: int, include_empty: bool = False) -> list[Candidate]:
    result = []
    for match in BOX.finditer(text, start, end):
        closing = matching_brace(text[:end], match.end() - 1)
        if closing is None:
            continue
        value = text[match.end():closing - 1].strip()
        if value or include_empty:
            result.append(Candidate(match.start(), closing, value, "框答"))
    return result


def grouped_boxes(text: str, start: int, end: int) -> list[Candidate]:
    result: list[Candidate] = []
    for box in boxes(text, start, end):
        if result:
            prev = result[-1]
            join = text[prev.end:box.start]
            if BOX_JOIN.fullmatch(join):
                separator = " or " if "or" in join else ", "
                result[-1] = Candidate(prev.start, box.end, prev.value + separator + box.value, "框答")
                continue
        result.append(box)
    return result


def thought_candidates(text: str, start: int, end: int) -> list[Candidate]:
    explicit = grouped_boxes(text, start, end)
    phrased = []
    for match in ANSWER.finditer(text, start, end):
        if CONDITIONAL.search(text[max(start, match.start() - 30):match.start()]):
            continue
        right = min(end, match.end() + PHRASE_LIMIT)
        segment = text[match.end():right].lstrip(" \t*:")
        left = right - len(segment)
        stop = ANSWER_STOP.search(segment)
        if stop is None and (right < end or right == len(text)):
            # 到预算上限仍未遇到边界的明示短语可能尚未说完。
            continue
        raw = segment[:stop.start() if stop else len(segment)].rstrip(" \t,*:")
        value = ANSWER_HEDGE.sub("", raw).strip()
        if not value or NON_VALUE.match(value):
            continue
        contained = [box for box in explicit if left <= box.start < left + len(raw)]
        if contained:
            # 同一短语中的框答保留结构和括号，避免同一候选重复入表。
            continue
        if BOX.search(value):
            continue
        phrased.append(Candidate(left, left + len(raw), value, "明示短语"))
    for match in ANSWER_BACKWARD.finditer(text, start, end):
        value = match["value"].strip()
        if not (NON_VALUE.match(value) or ANSWER_STOP.search(value) or BOX.search(value)):
            phrased.append(Candidate(match.start("value"), match.end(), value, "明示短语"))
    return sorted(explicit + phrased, key=lambda candidate: (candidate.end, candidate.start))


def normalize_phrase(value: str) -> str:
    value = value.translate(UNICODE_MATH).replace("$", "")
    value = re.sub(r"\\[()\[\]]", "", value)
    value = re.sub(r"√\s*(\d+(?:\.\d+)?|[a-zA-Z])", r"\\sqrt{\1}", value)
    value = DIGIT_GROUP.sub(lambda match: match.group().replace(",", ""), value)
    return value.strip()


@cache
def parse_gold(gold: str):
    return parse("$" + gold + "$")


@cache
def equivalent(value: str, gold: str) -> bool:
    # 与 common.py 的 LaTeX 包装及 Math-Verify 判等调用相同。
    try:
        return bool(verify(parse_gold(gold), parse(r"$\boxed{" + value + "}$")))
    except Exception as exc:
        print(f"判等异常，按原评分器计为未匹配 {value!r}: {exc}", file=sys.stderr)
        return False


@cache
def local_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def tokenizer_path(path: Path) -> Path | None:
    model = re.match(r"mech_([^_]+)_", path.name)
    repo = TOKENIZERS.get(model.group(1) if model else "1.7b")
    paths = sorted((HF / repo).glob("snapshots/*/tokenizer.json")) if repo else []
    return paths[0] if paths else None


def token_position(text: str, candidate_end: int, ids: list[int], tokenizer: Tokenizer) -> int:
    prefix = text[:candidate_end]
    low, high = 0, len(ids)
    while low < high:
        middle = (low + high) // 2
        decoded = tokenizer.decode(ids[:middle], skip_special_tokens=False)
        if decoded.startswith(prefix):
            high = middle
        else:
            low = middle + 1
    return low


def longest_tail_repeat(tokens: list) -> tuple[int, int, int]:
    offset = max(0, len(tokens) - REPEAT_TAIL)
    tail = tokens[offset:]
    first_matches: dict[tuple, int] = {}
    best = (0, 0, 0)
    for i in range(len(tail) - REPEAT_LENGTH + 1):
        key = tuple(tail[i:i + REPEAT_LENGTH])
        first = first_matches.setdefault(key, i)
        if i < len(tail) - REPEAT_RECENT or i - first < REPEAT_LENGTH:
            continue
        length = REPEAT_LENGTH
        while i + length < len(tail) and length < i - first and tail[first + length] == tail[i + length]:
            length += 1
        best = max(best, (length, offset + first, offset + i))
    return best


def excerpt_text(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def analyze_trace(row: dict, gold: str, tokenizer: Tokenizer | None, path: Path) -> Trace:
    text = row["text"]
    start = text.find("<think>") + len("<think>") if "<think>" in text else 0
    close = text.find("</think>", start)
    closed = close >= 0
    end = close if closed else len(text)
    candidates = thought_candidates(text, start, end)
    first = candidates[0] if candidates else None
    found_correct = next((item for item in candidates if equivalent(
        normalize_phrase(item.value) if item.source == "明示短语" else item.value, gold
    )), None)
    correct_box = any(equivalent(box.value, gold) for box in boxes(text, start, end))
    visible = boxes(text, close + len("</think>"), len(text), include_empty=True) if closed else []

    ids = row.get("out_ids")
    exact = bool(tokenizer and ids and len(ids) == row["ntok"]
                 and tokenizer.decode(ids, skip_special_tokens=False) == text)
    encoding = None
    if exact:
        tokens = ids
        token_source = "原始 ID"
    else:
        # 旧缓存可以没有 ID。分类仍可复现，分词来源及位置精度单独注明。
        encoding = tokenizer.encode(text, add_special_tokens=False) if tokenizer else None
        tokens = encoding.ids if encoding else SURFACE.findall(text.lower())
        token_source = "重分词估计" if encoding else "词面回退"
    first_token = None
    if first and exact:
        first_token = token_position(text, first.end, ids, tokenizer)
    elif first and encoding:
        first_token = next((i for i, (_, end_offset) in enumerate(encoding.offsets, 1)
                            if end_offset >= first.end), None)

    tail = text[-COMPRESS_CHARS:].encode()
    ratio = len(zlib.compress(tail)) / len(tail) if tail else 1.0
    repeat, earlier, later = longest_tail_repeat(tokens)
    observed_check = CHECK.search(text, max(first.end, end - CHECK_TAIL), end - CHECK_FOLLOW) if first and not closed else None
    if ratio < COMPRESS_RATIO or repeat:
        category = CLASSES[0]
        reason = "低压缩率" if ratio < COMPRESS_RATIO else "末尾块重复"
        if repeat and tokenizer:
            excerpt = " … ".join(excerpt_text(tokenizer.decode(tokens[i:i + min(repeat, 48)],
                                                                skip_special_tokens=False), 172)
                                 for i in (earlier, later))
        else:
            excerpt = excerpt_text(text[-360:])
    elif closed:
        category, reason = CLASSES[3], "已结束思考，输出仍被截断"
        excerpt = excerpt_text("</think> " + text[close + len("</think>"):])
    elif observed_check:
        category, reason = CLASSES[1], "候选后有尾部验算提示"
        excerpt = excerpt_text(text[max(start, observed_check.start() - 85):observed_check.start() + 240])
    elif not first:
        category, reason = CLASSES[2], "无明示候选及强重复信号"
        excerpt = excerpt_text(text[-360:])
    else:
        category, reason = CLASSES[3], "有候选，尾部缺少验算提示"
        excerpt = excerpt_text(text[max(start, first.start - 40):first.end + 230])

    return Trace(
        row=row, path=path, category=category, reason=reason, first=first,
        first_correct=first is not None and first == found_correct,
        found_correct=found_correct, correct_box=correct_box, has_visible_box=bool(visible),
        visible_correct_box=bool(visible and equivalent(visible[-1].value, gold)),
        closed=closed, first_token=first_token,
        token_denominator=len(tokens) if exact or encoding else None, token_source=token_source,
        zlib_ratio=ratio, repeat_length=repeat, excerpt=excerpt,
    )


def load_records(path: Path, *, allow_empty: bool = False) -> list[dict]:
    # pickle 输入须来自本实验或其他可信本地来源。
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or (not data and not allow_empty) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} 的记录列表格式不符")
    return data


def fraction(n: int, d: int) -> str:
    return f"{n}/{d}（{n / d:.1%}）" if d else "—"


def points(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def literal(text: str) -> str:
    return "`" + text.replace("|", r"\|").replace("\n", " ").replace("`", "&#96;") + "`"


def position(trace: Trace) -> str:
    if trace.first_token is None:
        return "不确定" if trace.first else "—"
    ntok = trace.token_denominator
    approximate = "约 " if trace.token_source == "重分词估计" else ""
    return f"{approximate}{trace.first_token}/{ntok}（{trace.first_token / ntok:.1%}）"


def continuation_path(paths: list[Path]) -> Path | None:
    matches = [re.fullmatch(r"(mech_[^_]+)_eval_base(?:\.part\d+)?\.pkl", path.name) for path in paths]
    if all(matches) and len({(path.parent, match.group(1)) for path, match in zip(paths, matches)}) == 1:
        return paths[0].with_name(matches[0].group(1) + "_cont.pkl")
    return None


def report_continuation(path: Path | None, rows: list[dict], traces: list[Trace]) -> None:
    if path is None or not path.is_file():
        if path is not None:
            print(f"- 本次读取时尚无 {literal(relative(path))}。")
        return
    ids = {row["pid"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("续写对照所用基线必须每题只有一条轨迹")
    truncated = {trace.row["pid"]: trace for trace in traces}
    data = load_records(path, allow_empty=True)
    by_pid = {row["pid"]: row for row in data}
    if len(by_pid) != len(data) or set(by_pid) & ids != set(truncated):
        raise ValueError(f"{path} 的 PID 与当前基线截断轨迹不匹配")
    if any(row["correct"] and not row["finished"] for row in data):
        raise ValueError(f"{path} 含未答完却判对的记录")
    finished = sum(by_pid[pid]["finished"] for pid in truncated)
    ok = sum(by_pid[pid]["correct"] for pid in truncated)
    both = sum(bool(trace.found_correct and by_pid[pid]["correct"]) for pid, trace in truncated.items())
    early = sum(trace.found_correct is not None for trace in traces)
    base = sum(row["correct"] for row in rows)
    print(f"- 2 倍预算续写在这些基线截断题中完成 {fraction(finished, len(traces))}，答对 "
          f"{fraction(ok, len(traces))}，总体增加 **{points(ok, len(rows))} 个百分点**至 "
          f"{(base + ok) / len(rows):.1%}。与思考候选估计相比，两者都挽回 {both} 道，"
          f"只有候选估计覆盖 {early - both} 道，只有续写答对 {ok - both} 道。"
          f"数据来自 {literal(relative(path))}，按同一基线的 PID 配对。"
          "续写保留原前缀，用种子 3000 再抽样至多 4096 token 后重评完整文本。[续写实现](../../mech.py#L239)")


def report_group(paths: list[Path], rows: list[dict], traces: list[Trace], cont: Path | None,
                 examples: dict[str, list[Trace]]) -> None:
    total, cut = len(rows), len(traces)
    right = sum(row["correct"] for row in rows)
    ok = sum(trace.found_correct is not None for trace in traces)
    first_ok = sum(trace.first_correct for trace in traces)
    boxed_ok = sum(trace.correct_box for trace in traces)
    visible_boxes = sum(trace.has_visible_box for trace in traces)
    visible_ok = sum(trace.visible_correct_box for trace in traces)
    closed = sum(trace.closed for trace in traces)
    print("\n## " + " + ".join(literal(path.name) for path in paths))
    print(f"\n基线答对 **{fraction(right, total)}**，截断 **{fraction(cut, total)}**。")
    position_rows = [trace for trace in traces if trace.first_token is not None and trace.token_source == "原始 ID"]
    counts = Counter(trace.category for trace in traces)
    if cut:
        print(f"截断时，{counts['仍在推进']} 条未检出明示候选或强重复，{counts['反复验算']} 条在候选后继续核查，"
              f"{counts['循环']} 条命中循环规则。{fraction(closed, cut)} 已输出 `</think>`，"
              f"其中 {visible_boxes} 条在输出区写出完整框，最后一框判对 {visible_ok} 条。")
    for label in CLASSES:
        options = [trace for trace in traces if trace.category == label]
        if options:
            picked = (max(options, key=lambda trace: trace.repeat_length) if label == CLASSES[0] else
                      next((trace for trace in options if (trace.first_correct and label == CLASSES[1])
                            or (label == CLASSES[2] and not CHECK.search(trace.row["text"][-500:]))
                            or (trace.closed and trace.found_correct and label == CLASSES[3])), options[0]))
            examples[label].append(picked)
    print(f"\n- 在已保存的截断轨迹中，思考里至少一个候选判对 **{fraction(ok, cut)}**。"
          f"若能识别并将每条轨迹中的正确候选写成最终框答，同时保持原有答完题的结果，"
          f"最多可从这些检出的候选挽回 **{points(ok, total)} 个百分点**"
          f"（{ok}/{total}，准确率升至 {(right + ok) / total:.1%}）。")
    print(f"- 只取首个候选，判对 {fraction(first_ok, cut)}，可挽回 {points(first_ok, total)} 个百分点。"
          f"只统计思考中逐个完整框答，至少一框判对 {fraction(boxed_ok, cut)}，折合 {points(boxed_ok, total)} 个百分点。"
          f"若截断题全部变对，形式上的最高增益为 {points(cut, total)} 个百分点。")
    if position_rows:
        q1, median, q3 = np.percentile([trace.first_token / trace.row["ntok"] for trace in position_rows], [25, 50, 75])
        print(f"- {len(position_rows)} 条首候选可按原始 ID 定位，位置中位数为预算的 {median:.1%}"
              f"（四分位区间 {q1:.1%}–{q3:.1%}）。")
    report_continuation(cont, rows, traces)
    print("\n### 截断分类和正确候选\n")
    print("| Level | 全部 | 截断 / 全部 | 循环 | 反复验算 | 仍在推进 | 其他 | 任一候选正确 | 首候选正确 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    levels = sorted({row["level"] for row in rows})
    for level in ["合计", *levels]:
        selected = traces if level == "合计" else [trace for trace in traces if trace.row["level"] == level]
        n = total if level == "合计" else sum(row["level"] == level for row in rows)
        c = len(selected)
        kinds = counts if level == "合计" else Counter(trace.category for trace in selected)
        correct = sum(trace.found_correct is not None for trace in selected)
        first_correct = sum(trace.first_correct for trace in selected)
        cells = [level, n, fraction(c, n), *(fraction(kinds[label], c) for label in CLASSES),
                 fraction(correct, c), fraction(first_correct, c)]
        print("| " + " | ".join(map(str, cells)) + " |")
    print("\n类别和候选列均以该层截断数为分母。\n")
    print("### 答完轨迹的长度\n")
    print("| Level | 答完 / 全部 | P10 | P25 | P50 | P75 | P90 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    medians = []
    for level in ["合计", *levels]:
        subset = rows if level == "合计" else [row for row in rows if row["level"] == level]
        lengths = [row["ntok"] for row in subset if not row["truncated"]]
        quantiles = np.percentile(lengths, [10, 25, 50, 75, 90]) if lengths else None
        values = " | ".join(f"{n:.1f}".removesuffix(".0") for n in quantiles) if lengths else " | ".join(["—"] * 5)
        if level != "合计" and lengths:
            medians.append((level, quantiles[2]))
        print(f"| {level} | {fraction(len(lengths), len(subset))} | {values} |")
    if len(medians) >= 2:
        trend = "逐层上升" if all(a[1] < b[1] for a, b in zip(medians, medians[1:])) else "随 level 有起伏"
        print(f"\n已答完轨迹的长度中位数{trend}，从 level {medians[0][0]} 的 {medians[0][1]:g} token "
              f"到 level {medians[-1][0]} 的 {medians[-1][1]:g} token。长度分布受预算筛选，尤其是完成率低的层。")
    stored = [trace.row["looping"] for trace in traces if "looping" in trace.row]
    print(f"\n原始 `looping` 字段覆盖 {len(stored)}/{cut} 条截断轨迹，其中 {sum(stored)} 条为真。"
          f"逐条复算的 6,000 字符压缩率规则命中 {sum(trace.zlib_ratio < COMPRESS_RATIO for trace in traces)} 条，"
          f"末尾块重复规则命中 {sum(bool(trace.repeat_length) for trace in traces)} 条。")

def report_rows(name: str, traces: list[Trace]) -> None:
    print(f"\n<details>\n<summary>{name} · {len(traces)} 条截断轨迹</summary>\n")
    print("| PID | Level | 分类 | 判据 | 首候选（来源） | 首次完整出现 | 首候选正确 | 首个判对的候选 | zlib 比 | 重复长度 / 分词 |")
    print("|---|---:|---|---|---|---|---|---|---:|---|")
    for trace in traces:
        first = trace.first
        found = trace.found_correct
        cells = [
            literal(trace.row["pid"]), str(trace.row["level"]), trace.category, trace.reason,
            f"{literal(first.value)}（{first.source}）" if first else "—",
            position(trace), ("是" if trace.first_correct else "否") if first else "—",
            f"{literal(found.value)}（{found.source}）" if found else "—", f"{trace.zlib_ratio:.3f}",
            f"{trace.repeat_length} / {trace.token_source}",
        ]
        print("| " + " | ".join(cells) + " |")
    print("\n</details>")


def method(data_path: Path, digest: str) -> None:
    print("# T11　Qwen 截断轨迹解剖")
    print(f"""
复跑命令为 `../.venv/bin/python -I -B handoff/codex/truncation_anatomy.py`。
位置参数可换成任意同结构的 pickle，来自同一实验臂的分片可加 `--combine` 合并统计。

答案按 `pid = unique_id` 对齐本地 MATH-500 缓存 {literal(data_path.parent.name + "/" + data_path.name)}。
文件 SHA-256 为 `{digest}`。读取该 JSONL 是因为 `common.py` 在导入时加载 MLX。
原始准确率直接采用 pickle 中的 `correct`，并按各文件分别计算。[数据加载](../../common.py#L4) · [评分协议](../../common.py#L77)

四类按下列顺序判定。位置以完整候选第一次可见时所用的生成 token 数除以 `ntok`，
反向明示短语在完整提示词出现时才计入。优先用保存的 `out_ids` 和本地 tokenizer 解码定位。
旧缓存若只有文本，重分词只提供 token 位置估计，
缺少可用 tokenizer 时只做词面分类。

1. **循环**。最后 {COMPRESS_CHARS:,} 字符的 UTF-8 字节经 zlib 压缩后小于原大小的 {COMPRESS_RATIO:.0%}，
   或最后 {REPEAT_TAIL:,} 个 token 中有两段非重叠的逐 token 相同片段，长至少 {REPEAT_LENGTH} token，
   后一段始于最后 {REPEAT_RECENT} token 内。后一规则也标出局部重复。没有匹配分词器时改用英文字母串、数字串或单个非空白字符作词面单位。
   [项目原循环规则](../../agent8b.py#L169)
2. **其他（已结束思考）**。存在 `</think>`，输出区仍在生成。按实际截断阶段归类，即使此前有验算。
3. **反复验算**。还在思考，已出现明示候选，候选之后的最后 {CHECK_TAIL:,} 字符内出现
   `let me check`、`double-check`、`another method` 等复核或换方法提示，且提示后至少有 {CHECK_FOLLOW} 字符。
4. **仍在推进**。未命中循环，还在思考，未检出候选。**其他（有候选）**涵盖有候选但没有上述尾部验算信号的轨迹。

候选识别限于思考区，从 `<think>` 后开始，遇 `</think>` 停止。无起始标记时从文本开头读取。
提取括号成对的非空 `\\boxed{{…}}`（相邻且用逗号或 and/or 相连的框合并）及 `the answer is …`、
`the final answer should be …`、`I think X is the correct answer` 等明示短语。
正向短语至多看 {PHRASE_LIMIT} 字符，在换行、句末符号或指定转折词处停止，
过滤 if/whether 转述和无值表述。短语统一简单的 Unicode 数学符号、千位分隔符和数学定界符。
按候选写完位置排序。候选判等使用项目的 Math-Verify 调用。表中“否”表示该候选未被评分器判为等价，
包括解析局限。纯隐式答案、未列出的提示词和数学记法会漏检。局部重复也可能来自展示同一推导，
分类仅描述可识别的文本信号。[判等实现](../../common.py#L82)""".rstrip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="可信轨迹 pkl，一文件一组")
    parser.add_argument("--combine", action="store_true", help="合并同一实验臂的互不重叠分片")
    parser.add_argument("--data", type=Path, help="指定本地 MATH-500 test.jsonl")
    parser.add_argument("--tokenizer", type=Path, help="覆盖本地 tokenizer.json")
    parser.add_argument("--continuation", type=Path, help="指定单组基线对应的可信续写 pkl")
    args = parser.parse_args()
    if args.combine and not args.inputs:
        parser.error("--combine 需要显式指定同一实验臂的分片")
    args.inputs = args.inputs or DEFAULT_INPUTS
    if args.continuation and len(args.inputs) > 1 and not args.combine:
        parser.error("--continuation 需要单组基线；同一臂的分片可用 --combine")
    if args.continuation and not args.continuation.is_file():
        parser.error(f"找不到续写文件 {args.continuation}")
    data_paths = sorted(HF.glob("datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl"))
    if not args.data and len(data_paths) != 1:
        parser.error("请用 --data 指定实验所用的本地 MATH-500 test.jsonl")
    data_path = (args.data or data_paths[0]).expanduser().absolute()
    contents = data_path.read_bytes()
    gold = {row["unique_id"]: row["answer"] for row in map(json.loads, contents.splitlines())}
    method(data_path, hashlib.sha256(contents).hexdigest())
    examples: dict[str, list[Trace]] = {label: [] for label in CLASSES}
    paths = [path.expanduser().resolve() for path in args.inputs]
    groups = [paths] if args.combine else [[path] for path in paths]
    audit = []
    for group in groups:
        rows, traces = [], []
        for path in group:
            batch = load_records(path)
            rows.extend(batch)
            tok_path = args.tokenizer or tokenizer_path(path)
            tokenizer = local_tokenizer(tok_path.expanduser().resolve()) if tok_path else None
            traces.extend(analyze_trace(row, gold[row["pid"]], tokenizer, path) for row in batch if row["truncated"])
        if args.combine and len({(row["pid"], row.get("sample", 0)) for row in rows}) != len(rows):
            parser.error("--combine 收到了重复的 (pid, sample)，请将不同实验臂分别输入")
        if any(trace.row["correct"] for trace in traces):
            parser.error("输入包含标作正确的截断轨迹，与本实验评分口径冲突")
        report_group(group, rows, traces, args.continuation or continuation_path(group), examples)
        audit.append((" + ".join(path.name for path in group), traces))
    print("\n## 轨迹短摘录\n")
    print("通过文件名和 PID 可在源 pickle 中找到原文。空白已折叠，省略号表示截短或跳到重复片段。")
    for label, items in examples.items():
        print(f"\n### {label}\n")
        if not items:
            print("本次输入没有命中这一类的轨迹。")
        for trace in items[:2]:
            print(f"{literal(trace.path.name)} · {literal(trace.row['pid'])}（{trace.reason}）")
            print(f"\n````text\n{trace.excerpt}\n````")
    print("按项目评分，合规输出须在 `</think>` 后写最终框答。提前结束需要选出候选并写入输出区，"
          "尤其是思考已经结束却在输出区重述推导的轨迹。原实现取 `split(\"</think>\")[-1]`，"
          "因此缺失结束标签的文本有一条技术例外，思考框在这种情况下也可能被读作最终框。"
          "正确候选的挽回估计使用标准答案选框，属于保存前缀上的事后估计，"
          "实际权重编辑也可能改变候选出现的时机、正确性和最终输出。[评分实现](../../common.py#L77)")
    print("\n## 逐条审计\n")
    for name, traces in audit:
        report_rows(name, traces)


if __name__ == "__main__":
    logging.getLogger("math_verify").setLevel(logging.CRITICAL)
    main()
