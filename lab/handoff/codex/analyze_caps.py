#!/usr/bin/env python3
"""离线汇总各生成上限的留出题结果。只读取本实验产生的可信 pickle。"""

import argparse
import importlib.util
import json
import math
from numbers import Integral
from pathlib import Path
import pickle
import sys
from types import ModuleType


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "out"
DEFAULT_ARMS = "base,s8,s8rand,concise,reseed"
DEFAULT_CAPS = "4096,8192,16384"


def import_analysis():
    # analyze.py 只从 common 取 OUT。临时提供该常量，避免导入真实的生成模块。
    spec = importlib.util.spec_from_file_location("analyze", ROOT / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    common = ModuleType("common")
    common.OUT = str(OUT)
    old_common = sys.modules.get("common")
    old_path = sys.path[:]
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.modules["common"] = common
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        if old_common is None:
            del sys.modules["common"]
        else:
            sys.modules["common"] = old_common
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
    return module


analysis = import_analysis()


def read_records(path, cap, expected_n):
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError("需要非空记录列表")
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"题数 {len(rows)} 与清单的 {expected_n} 不符")

    by_pid = {}
    for i, row in enumerate(rows, 1):
        pid, ntok = row["pid"], row["ntok"]
        trunc, correct = row["truncated"], row["correct"]
        if not isinstance(pid, (str, Integral)) or isinstance(pid, bool) or pid == "":
            raise ValueError(f"第 {i} 行题号无效")
        if pid in by_pid:
            raise ValueError(f"重复题号 {pid}")
        if not isinstance(ntok, Integral) or isinstance(ntok, bool) or not 0 <= ntok <= cap:
            raise ValueError(f"{pid} 的 ntok 超出范围或类型错误")
        if "out_ids" in row and len(row["out_ids"]) != ntok:
            raise ValueError(f"{pid} 的 ntok 与 out_ids 长度不一致")
        if not all(isinstance(flag, (bool, analysis.np.bool_)) for flag in (trunc, correct)):
            raise ValueError(f"{pid} 的截断或正确标记类型错误")
        if trunc != (ntok == cap) or (trunc and correct):
            raise ValueError(f"{pid} 的计分标记与 {cap} token 上限不一致")
        by_pid[pid] = row
    return by_pid


def arm_stats(records):
    rows = list(records.values())
    n = len(rows)
    n_correct = sum(int(r["correct"]) for r in rows)
    n_truncated = sum(int(r["truncated"]) for r in rows)
    n_finished = n - n_truncated
    usage = analysis.tokens(rows)
    if not n_correct:
        usage["tokens_per_correct"] = None
    return dict(n=n, n_correct=n_correct, n_truncated=n_truncated, n_finished=n_finished,
                acc=n_correct / n, trunc=n_truncated / n,
                acc_finished=n_correct / n_finished if n_finished else None, tokens=usage)


def compare(reference, target, reference_rows, target_rows):
    if reference_rows.keys() != target_rows.keys():
        only_ref = len(reference_rows.keys() - target_rows.keys())
        only_target = len(target_rows.keys() - reference_rows.keys())
        raise ValueError(f"题号集合不同（仅 {reference} 有 {only_ref} 题，仅 {target} 有 {only_target} 题）")
    before = list(reference_rows.values())
    after = [target_rows[pid] for pid in reference_rows]
    transitions = analysis.transitions(before, after)
    matrix = transitions["matrix"]
    wins = matrix["T"]["C"] + matrix["W"]["C"]
    losses = matrix["C"]["T"] + matrix["C"]["W"]
    discordant = wins + losses
    ci95 = analysis.bootstrap_ci(before, after) if discordant else (0.0, 0.0)
    return dict(reference=reference, target=target, n=len(before), delta=(wins - losses) / len(before),
                ci95=list(ci95), wins=wins, losses=losses,
                p_sign_two_sided=analysis.binomtest(wins, discordant, 0.5).pvalue if discordant else 1.0,
                p_sign_one_sided=(analysis.binomtest(wins, discordant, 0.5, alternative="greater").pvalue
                                  if discordant else 1.0),
                transitions=transitions)


def collect(input_dir, model, arms, caps):
    prefix = f"mech_{model}"
    summary_path = input_dir / f"{prefix}_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        arm_files = summary["arm_files"]
        expected_n = summary.get("n_eval")
        if not isinstance(arm_files, dict):
            raise ValueError(f"{summary_path.name} 的 arm_files 应为字典")
        if expected_n is not None and (type(expected_n) is not int or expected_n <= 0):
            raise ValueError(f"{summary_path.name} 的 n_eval 应为正整数")
        summary_missing = False
    except FileNotFoundError:
        arm_files, expected_n, summary_missing = {}, None, True

    result = dict(model=model, arms=arms, caps=caps, summary_file=str(summary_path),
                  summary_missing=summary_missing, by_cap={}, gain_by_cap={arm: {} for arm in arms})
    for cap in caps:
        block = dict(arms={}, vs_base={}, direct=None, skipped={}, comparison_issues=[])
        cap_key = str(cap)
        result["by_cap"][cap_key] = block
        suffix = "" if cap == 4096 else f".cap{cap}"
        records = {}
        for arm in dict.fromkeys(["base", *arms]):
            stem = arm_files.get(arm)
            if summary_missing:
                block["skipped"][arm] = "缺少结果清单"
                continue
            if stem is None:
                block["skipped"][arm] = "arm_files 无映射"
                continue
            if not isinstance(stem, str) or not stem or Path(stem).name != stem:
                raise ValueError(f"{summary_path.name} 的 {arm} 文件名无效")
            path = input_dir / f"{prefix}_{stem}{suffix}.pkl"
            try:
                records[arm] = read_records(path, cap, expected_n)
            except FileNotFoundError:
                block["skipped"][arm] = f"缺少 {path.name}"
            except (EOFError, pickle.UnpicklingError, OSError, ValueError, KeyError, TypeError) as exc:
                block["skipped"][arm] = f"{path.name} 记录无效（{exc}）"

        for arm in arms:
            if arm in records:
                block["arms"][arm] = arm_stats(records[arm])
            if arm in records and "base" in records:
                try:
                    block["vs_base"][arm] = compare("base", arm, records["base"], records[arm])
                except ValueError as exc:
                    block["comparison_issues"].append(f"base → {arm}：{exc}")
            paired = block["vs_base"].get(arm)
            result["gain_by_cap"][arm][cap_key] = {k: paired[k] for k in ("delta", "ci95")} if paired else None

        if "s8rand" in records and "s8" in records:
            try:
                block["direct"] = compare("s8rand", "s8", records["s8rand"], records["s8"])
            except ValueError as exc:
                block["comparison_issues"].append(f"s8rand → s8：{exc}")
    return result


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def gain(comparison):
    if comparison is None:
        return "—"
    lo, hi = comparison["ci95"]
    return f"{comparison['delta'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"


def render(result):
    arms, caps = result["arms"], result["caps"]
    lines = [f"# {result['model']} 多预算分析", "",
             "同一上限内按题号配对。Δ 为目标臂减参照臂的准确率差，单位为百分点。"
             "95% CI 按题有放回重抽 10,000 次。单侧检验方向为目标臂更准。", "",
             "token 统计以记录的输出长度为准，纳入截断回答。每答对一题的 token 为总输出 token 除以答对题数。"
             "提示和续写重新预填的计算成本另计。各臂共享上限，实际输出长度由各臂决定。"
             "8192 和 16384 档使用保存的续写结果。", ""]
    if result["summary_missing"]:
        lines.extend([f"缺少结果清单 `{Path(result['summary_file']).name}`，所有格留空。", ""])
    lines.extend(["## 增益随预算变化", "",
                  "| 实验臂 | " + " | ".join(str(cap) for cap in caps) + " |",
                  "| --- |" + " --- |" * len(caps)])
    for arm in arms:
        gains = result["gain_by_cap"][arm]
        lines.append(f"| {arm} | " + " | ".join(gain(gains[str(cap)]) for cap in caps) + " |")

    if result["summary_missing"]:
        return "\n".join(lines) + "\n"

    for cap in caps:
        block = result["by_cap"][str(cap)]
        lines.extend(["", f"## {cap} token", ""])
        if block["arms"]:
            lines.extend(["| 实验臂 | 题数 | 准确率 | 截断率 | 完成回答准确率 | 平均 token | token / 答对题数 |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for arm, stats in block["arms"].items():
            usage = stats["tokens"]
            per_correct = usage["tokens_per_correct"]
            per_correct = "—" if per_correct is None else f"{per_correct:.1f}"
            lines.append(f"| {arm} | {stats['n']} | {percent(stats['acc'])} | {percent(stats['trunc'])} | "
                         f"{percent(stats['acc_finished'])} | {usage['mean']:.1f} | {per_correct} |")

        comparisons = list(block["vs_base"].values())
        if block["direct"] is not None:
            comparisons.append(block["direct"])
        if comparisons:
            lines.extend(["", "| 参照 → 目标 | 配对数 | Δ [95% CI] | 胜/负 | 双侧 p | 单侧 p | 经截断 | 经完成回答 |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for c in comparisons:
                t = c["transitions"]
                lines.append(f"| {c['reference']} → {c['target']} | {c['n']} | {gain(c)} | {c['wins']}/{c['losses']} | "
                             f"{c['p_sign_two_sided']:.4g} | {c['p_sign_one_sided']:.4g} | "
                             f"{t['via_truncation'] * 100:+.2f} | {t['via_finished_answers'] * 100:+.2f} |")
            lines.extend(["", "T = 截断，W = 完成但错误，C = 完成且正确。", "",
                          "| 参照 → 目标 | T→T | T→W | T→C | W→T | W→W | W→C | C→T | C→W | C→C |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for c in comparisons:
                matrix = c["transitions"]["matrix"]
                counts = " | ".join(str(matrix[a][b]) for a in "TWC" for b in "TWC")
                lines.append(f"| {c['reference']} → {c['target']} | {counts} |")
        if block["skipped"] or block["comparison_issues"]:
            lines.append("")
            for arm, reason in block["skipped"].items():
                lines.append(f"- {arm} 跳过。{reason}。")
            lines.extend(f"- {issue}。" for issue in block["comparison_issues"])
    return "\n".join(lines) + "\n"


def self_test():
    from tempfile import TemporaryDirectory

    pairs = [(a, b) for a in "TWC" for b in "TWC"] + [("T", "C"), ("T", "C"), ("W", "C")]

    def rows(states, cap):
        lengths = {"T": cap, "W": 100, "C": 200}
        return [dict(pid=f"p{i}", ntok=lengths[s], truncated=s == "T", correct=s == "C")
                for i, s in enumerate(states)]

    with TemporaryDirectory(prefix="analyze_caps_") as work:
        folder = Path(work)
        arms = ["base", "s8", "s8rand", "concise", "reseed"]
        caps = [4096, 8192, 16384]
        files = {arm: f"fixture_{arm}" for arm in arms}
        manifest = folder / "mech_fixture_summary.json"
        manifest.write_text(json.dumps(dict(arm_files=files, n_eval=len(pairs))), encoding="utf-8")

        def save(arm, records, suffix=""):
            with (folder / f"mech_fixture_{files[arm]}{suffix}.pkl").open("wb") as stream:
                pickle.dump(records, stream)

        base = rows([a for a, _ in pairs], 4096)
        s8 = rows([b for _, b in pairs], 4096)
        save("base", base)
        save("s8", s8[::-1])
        save("s8rand", s8)
        save("concise", rows("T" * len(pairs), 4096))
        save("reseed", [dict(row, pid="other") if i == 0 else row for i, row in enumerate(base)])
        save("s8", [dict(row, ntok=8192) if row["truncated"] else row for row in s8], ".cap8192")
        save("base", base, ".cap8192.test1")
        save("base", base, ".cap16384")  # 旧截断标记混入新上限
        base_16k = rows([a for a, _ in pairs], 16384)
        s8_16k = rows([b for _, b in pairs], 16384)
        save("s8", s8_16k, ".cap16384")
        save("s8rand", base_16k, ".cap16384")
        save("reseed", [base_16k[0], *base_16k[:-1]], ".cap16384")  # 重复题号

        result = collect(folder, "fixture", arms, caps)
        block = result["by_cap"]["4096"]
        b = block["arms"]["base"]
        c = block["vs_base"]["s8"]
        assert (b["acc"], b["trunc"], b["acc_finished"]) == (3 / 12, 5 / 12, 3 / 7)
        assert b["tokens"]["mean"] == 1790 and b["tokens"]["tokens_per_correct"] == 7160
        assert (c["delta"], c["wins"], c["losses"]) == (0.25, 5, 2)
        assert (c["p_sign_one_sided"], c["p_sign_two_sided"]) == (29 / 128, 58 / 128)
        assert c["ci95"] == [-1 / 6, 2 / 3]
        assert c["transitions"]["matrix"] == {"T": {"T": 1, "W": 1, "C": 3}, "W": {"T": 1, "W": 1, "C": 2},
                                               "C": {"T": 1, "W": 1, "C": 1}}
        assert math.isclose(c["transitions"]["via_truncation"], 1 / 6)
        assert math.isclose(c["transitions"]["via_finished_answers"], 1 / 12)
        assert block["direct"]["reference"] == "s8rand" and block["direct"]["p_sign_two_sided"] == 1
        assert block["direct"]["ci95"] == [0, 0]
        assert block["arms"]["concise"]["acc_finished"] is None
        assert block["arms"]["concise"]["tokens"]["tokens_per_correct"] is None
        assert "reseed" in block["arms"] and "reseed" not in block["vs_base"]
        assert "题号集合不同" in block["comparison_issues"][0]
        assert result["by_cap"]["8192"]["arms"]["s8"]["trunc"] == 3 / 12
        assert "base" in result["by_cap"]["8192"]["skipped"]
        assert "标记" in result["by_cap"]["16384"]["skipped"]["base"]
        assert "重复题号" in result["by_cap"]["16384"]["skipped"]["reseed"]
        assert result["by_cap"]["16384"]["direct"]["delta"] == 0.25
        assert result["gain_by_cap"]["s8"]["8192"] is None
        assert "s8rand → s8" in render(result) and "T→C" in render(result)
        assert json.loads(json.dumps(result, allow_nan=False))["model"] == "fixture"
        assert collect(folder, "absent", arms, caps)["summary_missing"]
    print("合成数据自测通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="例如 q35-4b")
    parser.add_argument("--arms", default=DEFAULT_ARMS, help="逗号分隔的实验臂")
    parser.add_argument("--caps", default=DEFAULT_CAPS, help="逗号分隔的 token 上限")
    parser.add_argument("--input-dir", type=Path, default=OUT, help="实验记录目录，默认 exp/out")
    parser.add_argument("--output-dir", type=Path, default=HERE, help="报告目录，默认 handoff/codex")
    parser.add_argument("--self-test", action="store_true", help="用合成记录自测")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model or Path(args.model).name != args.model or args.model in {".", ".."}:
        parser.error("需要 --model，例如 q35-4b")
    arms = [arm.strip() for arm in args.arms.split(",")]
    try:
        caps = [int(cap) for cap in args.caps.split(",")]
    except ValueError:
        parser.error("--caps 应为逗号分隔的正整数")
    if not all(arms) or len(set(arms)) != len(arms):
        parser.error("--arms 中的实验臂应非空且互异")
    if min(caps) < 4096 or len(set(caps)) != len(caps):
        parser.error("--caps 中的上限须互异且至少为 4096")
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(OUT.resolve()):
        parser.error("请指定 exp/out 以外的报告目录")
    try:
        result = collect(args.input_dir.resolve(), args.model, arms, caps)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"读取结果清单失败（{exc}）")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"analyze_caps_{args.model}"
    md = stem.parent / f"{stem.name}.md"
    js = stem.parent / f"{stem.name}.json"
    md.write_text(render(result), encoding="utf-8")
    js.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    if result["summary_missing"]:
        print(f"缺少结果清单 {result['summary_file']}", file=sys.stderr)
    print(f"{md}\n{js}")


if __name__ == "__main__":
    main()
