"""Unified non-saturated hard reasoning benchmarks & rigorous grading (AIME, AMC23, OlympiadBench, MATH-500)."""
import csv
import glob
import json
import os
import re
import zlib
import numpy as np
import pyarrow.parquet as pq
from math_verify import parse, verify

HF = os.path.expanduser("~/.cache/huggingface/hub")


def _snap(pattern):
    snaps = sorted(glob.glob(f"{HF}/{pattern}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"Dataset snapshot not found for {pattern}")
    return snaps[-1]


def _clean_aime_int(s):
    """Normalize an AIME integer answer (0..999) if possible."""
    if s is None:
        return None
    t = str(s).strip()
    t = re.sub(r"^\$+|\$+$", "", t).strip()
    t = re.sub(r"^\\text\{(.*)\}$", r"\1", t).strip()
    t = t.replace(",", "").strip()
    if re.fullmatch(r"[+-]?\d+", t):
        return str(int(t))
    return None


def load_aime_all():
    """Load AIME 1983-2025 with clean deduplicated temporal splits.

    - 1983-2023 from di-zhang-fdu/AIME_1983_2024
    - 2024 (all 30) from Maxwell-Jia/AIME_2024
    - 2025 (all 30) from opencompass/AIME2025
    """
    rows = []
    # 1983-2023
    csv_path = os.path.join(_snap("datasets--di-zhang-fdu--AIME_1983_2024"), "AIME_Dataset_1983_2024.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            yr = int(r["Year"])
            if yr >= 2024:
                continue
            q = (r["Question"] or "").strip()
            a = _clean_aime_int(r["Answer"])
            if not q or a is None:
                continue
            rows.append(dict(
                pid=f"aime_{r['ID']}",
                source="aime",
                year=yr,
                problem=q,
                answer=a,
                level=int(r["Problem Number"]) if str(r.get("Problem Number", "")).isdigit() else 0,
            ))

    # 2024 (full 30 problems)
    p24 = glob.glob(os.path.join(_snap("datasets--Maxwell-Jia--AIME_2024"), "*.parquet"))[0]
    for r in pq.read_table(p24).to_pylist():
        a = _clean_aime_int(r["Answer"])
        rows.append(dict(
            pid=f"aime24_{r['ID']}",
            source="aime",
            year=2024,
            problem=str(r["Problem"]).strip(),
            answer=a if a is not None else str(r["Answer"]).strip(),
            level=0,
        ))

    # 2025 (full 30 problems: I and II)
    s25 = _snap("datasets--opencompass--AIME2025")
    for part in ("I", "II"):
        fp = os.path.join(s25, f"aime2025-{part}.jsonl")
        for idx, line in enumerate(open(fp, encoding="utf-8"), 1):
            r = json.loads(line)
            a = _clean_aime_int(r["answer"])
            rows.append(dict(
                pid=f"aime25_{part}_{idx}",
                source="aime",
                year=2025,
                problem=str(r["question"]).strip(),
                answer=a if a is not None else str(r["answer"]).strip(),
                level=idx,
            ))
    return rows


def load_amc23():
    fp = glob.glob(os.path.join(_snap("datasets--zwhe99--amc23"), "data", "*.parquet"))[0]
    out = []
    for r in pq.read_table(fp).to_pylist():
        ans = r["answer"]
        if isinstance(ans, float) and ans.is_integer():
            ans = str(int(ans))
        else:
            ans = str(ans).strip()
        out.append(dict(
            pid=f"amc23_{r['id']}",
            source="amc23",
            year=2023,
            problem=str(r["question"]).strip(),
            answer=ans,
            level=0,
        ))
    return out


def load_olympiad(subject="maths"):
    sub = "OE_TO_maths_en_COMP" if subject == "maths" else "OE_TO_physics_en_COMP"
    fp = glob.glob(os.path.join(_snap("datasets--Hothan--OlympiadBench"), "OlympiadBench", sub, "*.parquet"))[0]
    out = []
    for r in pq.read_table(fp).to_pylist():
        if r.get("is_multiple_answer"):
            continue
        fa = r.get("final_answer")
        if not fa or len(fa) != 1:
            continue
        ans = str(fa[0]).strip()
        if not ans:
            continue
        q = str(r["question"]).strip()
        if r.get("context"):
            q = str(r["context"]).strip() + "\n\n" + q
        unit = r.get("unit")
        if unit and str(unit).strip().lower() not in ("none", ""):
            q += f"\n(Express your final answer in units of {unit}.)"
        out.append(dict(
            pid=f"oly_{subject}_{r['id']}",
            source=f"oly_{subject}",
            year=0,
            problem=q,
            answer=ans,
            level=0,
            error=r.get("error"),
            answer_type=r.get("answer_type"),
        ))
    return out


def load_math500():
    fp = glob.glob(f"{HF}/datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl")[0]
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    ordered = [dict(
        pid=rows[i]["unique_id"],
        source="math500",
        year=0,
        problem=rows[i]["problem"],
        answer=rows[i]["answer"],
        level=int(rows[i].get("level", 0)),
    ) for i in idx]
    return ordered[:100], ordered[100:]


def _format_mc(question, options):
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question.strip()}\n\nOptions:\n{opts}\nPut only the single capital letter (A-{letters[len(options)-1]}) inside \\boxed{{}}."


def load_mmlupro_stem():
    fp = glob.glob(os.path.join(_snap("datasets--TIGER-Lab--MMLU-Pro"), "data", "test*.parquet"))[0]
    rows = [r for r in pq.read_table(fp).to_pylist() if r.get("category") in ("math", "physics", "chemistry")]
    rng = np.random.default_rng(101)
    idx = rng.permutation(len(rows))
    out = []
    for i in idx:
        r = rows[i]
        out.append(dict(
            pid=f"mmlupro_{r['question_id']}",
            source="mc",
            year=0,
            category=r["category"],
            problem=_format_mc(r["question"], r["options"]),
            answer=str(r["answer"]).strip().upper(),
            level=0,
        ))
    return out


def load_supergpqa_sci_hard():
    fp = os.path.join(_snap("datasets--m-a-p--SuperGPQA"), "SuperGPQA-all.jsonl")
    rows = []
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("discipline") == "Science" and r.get("difficulty") == "hard":
            rows.append(r)
    rng = np.random.default_rng(202)
    idx = rng.permutation(len(rows))
    out = []
    for i in idx:
        r = rows[i]
        out.append(dict(
            pid=f"sgpqa_{r['uuid'][:12]}",
            source="mc",
            year=0,
            category=r.get("field", "Science"),
            problem=_format_mc(r["question"], r["options"]),
            answer=str(r["answer_letter"]).strip().upper(),
            level=0,
        ))
    return out


def load_supergpqa_concept_hard():
    fp = os.path.join(_snap("datasets--m-a-p--SuperGPQA"), "SuperGPQA-all.jsonl")
    rows = []
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("discipline") in ("Science", "Engineering") and r.get("difficulty") == "hard" and not r.get("is_calculation"):
            rows.append(r)
    rng = np.random.default_rng(303)
    idx = rng.permutation(len(rows))
    out = []
    for i in idx:
        r = rows[i]
        out.append(dict(
            pid=f"sgpqac_{r['uuid'][:12]}",
            source="mc",
            year=0,
            category=r.get("field", r.get("discipline", "STEM")),
            problem=_format_mc(r["question"], r["options"]),
            answer=str(r["answer_letter"]).strip().upper(),
            level=0,
        ))
    return out


def load_gpqa_diamond():
    fp = os.path.join(_snap("datasets--fingertap--GPQA-Diamond"), "test", "gpqa_diamond.parquet")
    rows = pq.read_table(fp).to_pylist()
    out = []
    for i, r in enumerate(rows):
        q = str(r["question"]).strip() + "\nPut only the single capital letter (A, B, C, or D) inside \\boxed{}."
        out.append(dict(
            pid=f"gpqad_{i:03d}",
            source="mc",
            year=0,
            problem=q,
            answer=str(r["answer"]).strip().upper(),
            level=0,
        ))
    return out


def load_benchmark(name):
    """Unified benchmark selector with strict temporal / held-out isolation.

    Available splits:
      - aime_practice / aime_gate / aime_test / aime_24_25 / amc23
      - oly_math_practice / oly_math_eval / oly_math_all / oly_phys
      - math500_practice / math500_eval / math500_hard_eval
      - mmlupro_stem_practice (200) / mmlupro_stem_gate (200) / mmlupro_stem_test (400)
      - supergpqa_sci_hard_practice (200) / supergpqa_sci_hard_gate (200) / supergpqa_sci_hard_test (400)
      - gpqa_diamond (198)
    """
    if name.startswith("aime"):
        all_aime = load_aime_all()
        if name == "aime_practice":
            return [r for r in all_aime if 2010 <= r["year"] <= 2018]
        if name == "aime_gate":
            return [r for r in all_aime if 2019 <= r["year"] <= 2021]
        if name == "aime_test":
            return [r for r in all_aime if 2022 <= r["year"] <= 2025]
        if name == "aime_24_25":
            return [r for r in all_aime if r["year"] in (2024, 2025)]
        if name == "aime_all":
            return all_aime
    elif name == "amc23":
        return load_amc23()
    elif name.startswith("oly_math"):
        rows = load_olympiad("maths")
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(rows))
        shuf = [rows[i] for i in idx]
        if name == "oly_math_practice":
            return shuf[:120]
        if name == "oly_math_eval":
            return shuf[120:320]
        return shuf
    elif name == "oly_phys":
        return load_olympiad("physics")
    elif name.startswith("math500"):
        prac, ev = load_math500()
        if name == "math500_practice":
            return prac
        if name == "math500_eval":
            return ev
        if name == "math500_hard_eval":
            return [r for r in ev if r["level"] >= 4]
    elif name.startswith("mmlupro_stem"):
        all_m = load_mmlupro_stem()
        if name == "mmlupro_stem_practice":
            return all_m[:200]
        if name == "mmlupro_stem_gate":
            return all_m[200:400]
        if name == "mmlupro_stem_test":
            return all_m[400:800]
    elif name.startswith("supergpqa_sci_hard"):
        all_s = load_supergpqa_sci_hard()
        if name == "supergpqa_sci_hard_practice":
            return all_s[:200]
        if name == "supergpqa_sci_hard_gate":
            return all_s[200:400]
        if name == "supergpqa_sci_hard_test":
            return all_s[400:800]
    elif name.startswith("supergpqa_concept"):
        all_c = load_supergpqa_concept_hard()
        if name == "supergpqa_concept_practice":
            return all_c[:120]
        if name == "supergpqa_concept_gate":
            return all_c[120:240]
        if name == "supergpqa_concept_test":
            return all_c[240:]
    elif name == "gpqa_diamond":
        return load_gpqa_diamond()
    raise ValueError(f"Unknown benchmark split: {name}")


def last_boxed(s):
    """Content of the last balanced \\boxed{...} in s, or None."""
    i = s.rfind("\\boxed")
    while i >= 0:
        j = s.find("{", i)
        if j >= 0 and s[i + 6:j].strip() == "":
            depth = 0
            for k in range(j, len(s)):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[j + 1:k]
        i = s.rfind("\\boxed", 0, i)
    return None


def is_looping(text, truncated):
    """Detect degenerate repetition in truncated trajectories (compatible with agent8b.py)."""
    if not truncated or len(text) < 400:
        return False
    tail = text[-1200:]
    ratio = len(zlib.compress(tail.encode("utf-8", "ignore"))) / max(1, len(tail))
    if ratio < 0.22:
        return True
    words = tail.split()
    if len(words) >= 80:
        grams = [" ".join(words[i:i + 8]) for i in range(len(words) - 7)]
        if len(set(grams)) < 0.35 * len(grams):
            return True
    return False


def _verify_pair(gold, pred, source="math", error=None):
    if pred is None:
        return False
    # Exact multiple-choice letter match (handles \text{A}, (A), A., etc.)
    if source == "mc":
        g_let = str(gold).strip().upper()
        p_clean = re.sub(r"\\text\{([^}]*)\}|\\mathrm\{([^}]*)\}|\\mathbf\{([^}]*)\}", lambda m: next(x for x in m.groups() if x is not None), str(pred))
        p_clean = re.sub(r"[\$\(\)\[\]\.\s]", "", p_clean).upper()
        if p_clean == g_let:
            return True
        m = re.match(r"^([A-J])(?:[^A-Z]|$)", p_clean)
        return bool(m and m.group(1) == g_let)
    # Fast exact integer match for AIME
    if source == "aime":
        gi = _clean_aime_int(gold)
        pi = _clean_aime_int(pred)
        if gi is not None and pi is not None:
            return gi == pi
    # Strip outer $ if present in gold
    g_clean = re.sub(r"^\$+|\$+$", "", str(gold).strip()).strip()
    p_clean = re.sub(r"^\$+|\$+$", "", str(pred).strip()).strip()
    if g_clean == p_clean:
        return True
    # Numeric tolerance check for OlympiadBench physics/numerical answers
    if error is not None:
        try:
            g_val = float(g_clean)
            p_val = float(p_clean)
            tol = float(error) if not isinstance(error, (list, tuple)) else float(error[0])
            if abs(g_val - p_val) <= max(tol, 1e-4 * abs(g_val) + 1e-6):
                return True
        except Exception:
            pass
    try:
        return bool(verify(parse("$" + g_clean + "$"), parse("$\\boxed{" + p_clean + "}$")))
    except Exception:
        return False


def grade_response(text, ntok, max_tokens, problem):
    """Strict paper rule (truncated -> wrong) plus rich diagnostic breakdown for W->C analysis."""
    truncated = ntok >= max_tokens
    has_end_think = "</think>" in text
    after_think = text.split("</think>")[-1] if has_end_think else text
    boxed = last_boxed(after_think)
    if boxed is None and not has_end_think:
        # Check if model put boxed inside thinking before stopping
        boxed_any = last_boxed(text)
    else:
        boxed_any = boxed

    gold = problem["answer"]
    src = problem.get("source", "math")
    err = problem.get("error")

    boxed_ok = _verify_pair(gold, boxed, source=src, error=err) if boxed is not None else False
    boxed_any_ok = boxed_ok if boxed is not None else (
        _verify_pair(gold, boxed_any, source=src, error=err) if boxed_any is not None else False
    )

    correct = (not truncated) and boxed_ok
    finished_wrong = (not truncated) and (not correct)
    state = "T" if truncated else ("C" if correct else "W")

    think_part = text.split("</think>")[0] if has_end_think else text
    ans_part = text.split("</think>")[-1] if has_end_think else ""

    return dict(
        truncated=truncated,
        correct=correct,
        finished_wrong=finished_wrong,
        state=state,
        has_end_think=has_end_think,
        boxed=boxed,
        boxed_ok_ignoring_trunc=boxed_any_ok,
        looping=is_looping(text, truncated),
        ntok=int(ntok),
        think_chars=len(think_part),
        ans_chars=len(ans_part),
    )
