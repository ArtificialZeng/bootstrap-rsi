"""Assemble the public release folder (../release) from this working folder: code, small result files, reviews,
the report page. Raw answer files (*.pkl, *.jsonl, ~470 MB) and worker logs are left out. Local links in the
markdown files that point at files not in the release are turned into plain text; absolute paths are refused."""
import os, re, shutil, sys, glob

SRC = os.path.dirname(os.path.abspath(__file__))
REL = os.path.join(SRC, "..", "release")
DST = os.path.join(REL, "lab")
CODE = ["common.py", "mech.py", "ext.py", "cap_table.py", "analyze.py", "agent8b.py", "test_agent8b.py", "seam_check.py",
        "check_prefill.py", "run.py", "qrun.sh", "make_release.py"]
OUT = ["q35_caps.json", "q35_caps.md", "results.json", "results.md", "prereg_16384.md", "prereg_chain.md", "check_prefill_q35-4b.json",
       "mech_1.7b_summary.json", "mech_1.7b_config.json", "mech_q35-4b_summary.json", "mech_q35-4b_config.json", "seam_q35-4b.json"]
OUT_GLOBS = ["mech_q35-4b_eval_*.meta.json", "figs/*.svg"]
HANDOFF = ["review2.md", "review_agent8b.md", "TASK_review2.md", "TASK_review_agent8b.md", "codex/COMMON.md",
           "codex/make_figs.py", "codex/fig_caps.py", "codex/analyze_caps.py", "codex/analyze_sessions.py", "codex/truncation_anatomy.py",
           "codex/test_stats.py", "codex/figs_caps/*.svg"]
HANDOFF_GLOBS = [f"codex/T{i}_*.md" for i in (1, 2, 5, 6, 8, 11, 12, 15, 16)]  # reviews, analyses, current article
REPORT = ["build_page.py", "template.html", "curve.json", "curve.md", "index.html"]
SKIP = re.compile(r"\.(log|pkl|jsonl|npz)$|\.last\.txt$")


def cp(rel, base=""):
    for p in glob.glob(os.path.join(SRC, base, rel)):
        if SKIP.search(p) or os.path.isdir(p):
            continue
        d = os.path.join(DST, os.path.relpath(p, SRC))
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(p, d)


def main():
    shutil.rmtree(DST, ignore_errors=True)
    for f in CODE: cp(f)
    for f in OUT + OUT_GLOBS: cp(f, "out")
    for f in HANDOFF + HANDOFF_GLOBS: cp(f, "handoff")
    for f in REPORT: cp(f, "report")
    # the published page needs its own document skeleton when served by GitHub Pages
    page = open(os.path.join(SRC, "report", "index.html"), encoding="utf-8").read()
    os.makedirs(os.path.join(REL, "docs"), exist_ok=True)
    open(os.path.join(REL, "docs", "index.html"), "w", encoding="utf-8").write(
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body style="margin:0">\n' + page + "\n</body>\n</html>\n")
    # links: keep web links and links to files that exist in the release; everything else becomes plain text
    link = re.compile(r"\[([^\]\n]*)\]\((<[^>\n]*>|[^)\s]*)\)")
    dropped = 0
    for root, _, files in os.walk(REL):
        if ".git" in root.split(os.sep):
            continue
        for f in files:
            if not f.endswith(".md"):
                continue
            p = os.path.join(root, f); s = open(p, encoding="utf-8").read()
            def fix(m):
                nonlocal dropped
                t = m.group(2).strip("<>")
                if re.match(r"https?://|#|mailto:", t):
                    return m.group(0)
                target = os.path.normpath(os.path.join(root, t.split("#")[0]))
                if not t.startswith("/") and os.path.exists(target) and os.path.abspath(target).startswith(os.path.abspath(REL)):
                    return m.group(0)
                dropped += 1
                return m.group(1)
            s2 = link.sub(fix, s)
            if s2 != s:
                open(p, "w", encoding="utf-8").write(s2)
    # nothing private may ship
    bad = []
    pat = re.compile("|".join(["/Use" + "rs/", "scratch-" + "workspaces", "/private" + "/tmp", "@gm" + "ail", "whats" + "app"]), re.I)
    for root, _, files in os.walk(REL):
        if ".git" in root.split(os.sep):
            continue
        for f in files:
            p = os.path.join(root, f)
            try:
                for i, line in enumerate(open(p, encoding="utf-8"), 1):
                    if pat.search(line):
                        bad.append(f"{os.path.relpath(p, REL)}:{i}: {line.strip()[:140]}")
            except UnicodeDecodeError:
                pass
    n = sum(len(fs) for r, _, fs in os.walk(REL) if ".git" not in r.split(os.sep))
    print(f"{n} files, {dropped} local links turned into text, {len(bad)} lines to scrub")
    print("\n".join(bad[:40]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
