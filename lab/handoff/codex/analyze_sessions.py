#!/usr/bin/env python3
"""离线分析 agent8b 会话转录。模型生成的 Python 仅用 AST 检查。"""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
import sys


_CHECK = re.compile(r"^\[CHECK on reserved problems[^\]\n]*\]\s*", re.M)
_LITERAL_START = re.compile(r"^(?:\s*\[[a-zA-Z_ ]+\]\s+)?[^\[\]{}\n]{0,100}([\[{])")
_SESSION_FRAME = re.compile(r'File "<session>", line (\d+)')
_ERROR = re.compile(
    r"^(?P<kind>\w+(?:Error|Exception)|Error|Exception|StopIteration|StopAsyncIteration|"
    r"KeyboardInterrupt|SystemExit|GeneratorExit)"
    r"(?: \(session line (?P<line>\d+)\))?:\s*(?P<message>.*)$"
)


def _record(text):
    """Accept JSON plus printed Python containers, including NumPy's bare nan/inf."""
    try:
        value = json.loads(text)
    except ValueError:
        try:
            tree = ast.parse(text, mode="eval")
            class NonFinite(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.Constant(float(node.id)) if node.id in {"nan", "inf"} else node

            value = ast.literal_eval(NonFinite().visit(tree))
        except (SyntaxError, ValueError, TypeError, RecursionError):
            return None
    return value if isinstance(value, (dict, list)) else None


def split_feedback(text):
    """Return recorded stdout and the trailing framework check, when parseable."""
    for marker in reversed(list(_CHECK.finditer(text))):
        tail = text[marker.end():].split("\nNow call confirm()", 1)[0].strip()
        value = _record(tail)
        if isinstance(value, dict):
            return text[:marker.start()].rstrip("\n"), value
    return text, None


def _literal_end(text, start):
    closes = {"[": "]", "{": "}", "(": ")"}
    stack = []
    quote = None
    escaped = False
    for index, char in enumerate(text[start:], start):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in closes:
            stack.append(closes[char])
        elif char in "]})":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return index + 1
    return None


def printed_values(body):
    """Recover balanced, parseable dicts/lists, with 1-based output line numbers."""
    found = []
    offset = consumed = 0
    for line_number, line in enumerate(body.splitlines(keepends=True), 1):
        line_end = offset + len(line)
        while line_end > consumed and (start := _LITERAL_START.match(line[max(0, consumed - offset):])):
            start_offset = max(offset, consumed) + start.start(1)
            end_offset = _literal_end(body, start_offset)
            if end_offset is None:
                break
            value = _record(body[start_offset:end_offset])
            if value is None:
                break
            found.append((line_number, value))
            consumed = end_offset
        offset = line_end
    return found


def errors_in(body):
    """Return (output line, kind, message, session-code line) when evidenced."""
    found = []
    source_line = None
    for index, raw_line in enumerate(body.splitlines(), 1):
        line = raw_line.strip()
        if line.startswith("Traceback (most recent call last):"):
            source_line = None
        if frame := _SESSION_FRAME.search(line):
            source_line = int(frame[1])
        if match := _ERROR.fullmatch(line):
            message = match["message"]
            syntax_line = re.search(r"\(<session>, line (\d+)\)$", message)
            location = match["line"] or (syntax_line[1] if syntax_line else None)
            found.append((index, match["kind"], message, int(location) if location else source_line))
            source_line = None
        elif line.startswith("[BudgetError]"):
            found.append((index, "BudgetError", line[13:].strip(), None))
        elif line.startswith("[TimeoutError:"):
            found.append((index, "TimeoutError", line[14:].removesuffix("]").strip(), None))
        elif line.startswith("refused:"):
            found.append((index, "CodeRefused", line[8:].strip(), None))
        elif line.startswith("[Your ") and "token limit while thinking" in line:
            found.append((index, "ThinkingLimit", line.strip("[]"), None))
        elif line.startswith("[No ```python block found."):
            found.append((index, "MissingCode", line.strip("[]"), None))
        elif re.fullmatch(r"\.\.\. \[\d+ chars cut\] \.\.\.", line):
            found.append((index, "OutputTruncated", line, None))
    return found


TOOLS = set("""
    get_weight set_weight weaken reset current_edits answer_text reference problem
    write_notes run paired_test hidden_states submit confirm withdraw budget check
""".split())
FORWARD = {"run", "paired_test", "hidden_states"}
EDITS = {"set_weight", "weaken"}
DIAGNOSIS = FORWARD | {"answer_text", "reference", "problem"}
REPLY_ISSUES = {"ThinkingLimit", "MissingCode", "OutputTruncated"}
PAIR_FIELDS = {"delta", "base_acc", "edit_acc", "base_trunc", "edit_trunc", "wins", "losses"}
RESULT_FILE = Path(__file__).resolve().parents[2] / "out/8b_c4096_p60_e400_results.json"
UNKNOWN = object()
MAX_ITEMS = 256


def pid(value):
    return isinstance(value, str) or type(value) is int


def ids_of(value):
    if isinstance(value, (list, tuple)) and value and all(pid(i) for i in value):
        return tuple(value)
    return None


def correctness(value):
    return bool(value) if type(value) in (bool, int) and value in (0, 1) else None


@dataclass(frozen=True)
class Fact:
    value: object = UNKNOWN
    sources: frozenset = frozenset()
    projection: frozenset = frozenset()
    correct: bool | None = None
    outputs: frozenset = frozenset()


def combine(facts, value=UNKNOWN):
    facts = list(facts)
    return Fact(value, frozenset().union(*(f.sources for f in facts)),
                frozenset().union(*(f.projection for f in facts)),
                outputs=frozenset().union(*(f.outputs for f in facts)))


def qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = qualified(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def failure_filter(node, variable, record_variable=""):
    """Identify a filter on the cached baseline's per-question correct flag."""
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            and not isinstance(node.operand, ast.BoolOp)):
        flag = failure_filter(node.operand, variable, record_variable)
        return None if flag is None else not flag
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = node.left, node.comparators[0]
        flag = failure_filter(left, variable, record_variable) if not isinstance(left, ast.BoolOp) else None
        target = correctness(right.value) if isinstance(right, ast.Constant) else None
        if flag is not None and target is not None:
            if isinstance(node.ops[0], (ast.Eq, ast.Is)):
                return flag == target
            if isinstance(node.ops[0], (ast.NotEq, ast.IsNot)):
                return flag != target
        return None
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        flags = {failure_filter(child, variable, record_variable) for child in node.values} - {None}
        return next(iter(flags)) if len(flags) == 1 else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr != "get" or not node.args:
            return None
        record, key = node.func.value, node.args[0]
    elif isinstance(node, ast.Subscript):
        record, key = node.value, node.slice
    else:
        return None
    if not (isinstance(key, ast.Constant) and key.value == "correct"):
        return None
    if record_variable and qualified(record) == record_variable:
        return True
    if (isinstance(record, ast.Subscript) and qualified(record.value) == "base_results"
            and qualified(record.slice) == variable):
        return True
    return None


@dataclass
class Call:
    tool: str
    turn: int | None
    line: int | None = None
    ids: tuple | None = None
    correct: bool | None = None
    matrix: str | None = None
    direction: str | None = None
    which: str | None = None
    edited: bool | None = None
    partly_known: bool = False
    evidence: str = "代码"
    result: dict = field(default_factory=dict)
    failed: bool = False
    changes: dict = field(default_factory=dict)


def direction(fact):
    sources = fact.projection or fact.sources
    if "random" in sources and "hidden" in sources:
        return "其他（混合来源）"
    if "hidden_diff" in sources:
        return "隐藏状态差"
    if "random" in sources:
        return "随机"
    if "hidden" in sources:
        return "其他（隐状态，未识别到差分）"
    if "weight" in sources:
        return "权重本身"
    return "其他／不确定"


class CodeScan:
    """Limited constant propagation. Never calls code or reads its namespace at runtime."""

    def __init__(self, variables, turn, stop_line=None):
        self.variables = variables
        self.turn = turn
        self.stop_line = stop_line
        self.calls = []
        self.prints = []
        self.uncertain = False

    def value(self, node):
        if node is None:
            return Fact()
        if isinstance(node, ast.Constant):
            return Fact(node.value)
        if isinstance(node, ast.Name):
            return self.variables.get(node.id, Fact())
        if isinstance(node, ast.Subscript):
            subject, index = self.value(node.value), self.value(node.slice)
            payload = UNKNOWN
            if type(subject.value) in (str, tuple, list, dict) and index.value is not UNKNOWN:
                try:
                    payload = subject.value[index.value]
                except (IndexError, KeyError, TypeError, ValueError):
                    pass
            return replace(subject, value=payload)
        if isinstance(node, ast.Slice):
            parts = [self.value(p).value if p is not None else None
                     for p in (node.lower, node.upper, node.step)]
            return Fact(slice(*parts)) if all(p is None or type(p) is int for p in parts) else Fact()
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            parts = [self.value(p) for p in node.elts]
            values = tuple(p.value for p in parts)
            payload = values if all(v is not UNKNOWN for v in values) else UNKNOWN
            result = combine(parts, payload)
            flags = {p.correct for p in parts}
            return replace(result, correct=next(iter(flags)) if len(flags) == 1 else None)
        if isinstance(node, ast.Dict):
            parts = [(self.value(k), self.value(v)) for k, v in zip(node.keys, node.values)]
            try:
                payload = {k.value: v.value for k, v in parts}
            except TypeError:
                payload = UNKNOWN
            return combine([v for _, v in parts], payload)
        if isinstance(node, ast.BinOp):
            return self.binary(self.value(node.left), node.op, self.value(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = self.value(node.operand)
            payload = UNKNOWN
            if type(operand.value) in (int, float, bool):
                if isinstance(node.op, ast.USub):
                    payload = -operand.value
                elif isinstance(node.op, ast.Not):
                    payload = not operand.value
            return replace(operand, value=payload)
        if isinstance(node, ast.Attribute):
            subject = self.value(node.value)
            return Fact() if node.attr in {"shape", "size", "ndim", "dtype"} else replace(subject, value=UNKNOWN)
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            return self.comprehension(node)
        if isinstance(node, ast.Lambda):
            return Fact()
        if isinstance(node, (ast.IfExp, ast.BoolOp)):
            if isinstance(node, ast.IfExp):
                test = self.value(node.test)
                if type(test.value) is bool:
                    return self.value(node.body if test.value else node.orelse)
                children = (node.body, node.orelse)
            else:
                self.value(node.values[0])
                children = node.values[1:]
            was_uncertain = self.uncertain
            self.uncertain = True
            result = combine(self.value(child) for child in children)
            self.uncertain = was_uncertain
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.value(node.left), self.value(node.comparators[0])
            if (type(left.value) in (bool, int, float, str, type(None))
                    and type(right.value) in (bool, int, float, str, type(None))):
                if isinstance(node.ops[0], ast.Eq):
                    return Fact(left.value == right.value)
                if isinstance(node.ops[0], ast.NotEq):
                    return Fact(left.value != right.value)
            return combine([left, right])
        return combine(self.value(child) for child in ast.iter_child_nodes(node))

    @staticmethod
    def binary(left, op, right):
        payload = UNKNOWN
        if isinstance(op, ast.Add) and type(left.value) is type(right.value) in (str, tuple):
            if len(left.value) + len(right.value) <= MAX_ITEMS:
                payload = left.value + right.value
        result = combine([left, right], payload)
        if isinstance(op, ast.Sub) and all("hidden" in f.sources for f in (left, right)):
            result = replace(result, sources=result.sources | {"hidden_diff"})
        if isinstance(op, ast.Add) and left.correct == right.correct:
            result = replace(result, correct=left.correct)
        return result

    def comprehension(self, node):
        before = self.variables.copy()
        was_uncertain = self.uncertain
        self.uncertain = True
        for clause in node.generators:
            sequence = self.value(clause.iter)
            variable = qualified(clause.target)
            record_var = ""
            if (isinstance(clause.target, (ast.Tuple, ast.List)) and len(clause.target.elts) == 2
                    and isinstance(clause.iter, ast.Call) and qualified(clause.iter.func) == "base_results.items"):
                variable, record_var = map(qualified, clause.target.elts)
            flags = {failure_filter(test, variable, record_var) for test in clause.ifs} - {None}
            flag = next(iter(flags)) if len(flags) == 1 else sequence.correct
            self.bind(clause.target, replace(sequence, value=UNKNOWN))
            if variable:
                self.variables[variable] = replace(sequence, value=UNKNOWN, correct=flag)
            for test in clause.ifs:
                self.value(test)
        if isinstance(node, ast.DictComp):
            result = combine([self.value(node.key), self.value(node.value)])
        else:
            result = self.value(node.elt)
        self.variables.clear()
        self.variables.update(before)
        self.uncertain = was_uncertain
        return replace(result, value=UNKNOWN)

    def call(self, node):
        name = qualified(node.func)
        receiver = self.value(node.func.value) if isinstance(node.func, ast.Attribute) else Fact()
        args = [self.value(arg) for arg in node.args]
        kw = {k.arg: self.value(k.value) for k in node.keywords}

        def argument(index, name, default=UNKNOWN):
            return args[index] if index < len(args) else kw.get(name, Fact(default))

        if name in TOOLS - {"check"}:
            call = Call(name, self.turn, node.lineno, evidence="候选" if self.uncertain else "代码")
            if name in DIAGNOSIS:
                target = argument(0, "pid" if name in {"answer_text", "reference", "problem"} else "ids")
                call.ids = (target.value,) if pid(target.value) else ids_of(target.value)
                call.correct = target.correct
                if name == "hidden_states":
                    texts = argument(2, "texts", None).value
                    call.partly_known = texts is UNKNOWN or bool(texts)
                if name == "answer_text":
                    which = argument(1, "which", "base").value
                    call.which = which if isinstance(which, str) else None
                if name == "run":
                    edited = argument(1, "edited", True).value
                    call.edited = edited if type(edited) is bool else None
            if name in EDITS | {"get_weight", "reset"}:
                matrix = argument(0, "name").value
                call.matrix = matrix if isinstance(matrix, str) else None
            if name in EDITS:
                fact = argument(1, "u" if name == "weaken" else "W")
                call.direction = direction(fact)
            self.calls.append(call)
            if name in {"paired_test", "run"}:
                return Fact(outputs=frozenset({(name, self.turn, node.lineno)}))
            source = {"get_weight": "weight", "hidden_states": "hidden"}.get(name)
            return Fact(sources=frozenset({source})) if source else Fact()
        result = combine([receiver, *args, *kw.values()])
        if re.search(r"(?:^|\.)random(?:\.|$)", name) or name in {"default_rng", "RandomState"}:
            return replace(result, sources=result.sources | {"random"})
        if name.rsplit(".", 1)[-1] == "outer" and args:
            return replace(result, projection=args[0].sources)
        if name.rsplit(".", 1)[-1] == "subtract" and len(args) >= 2:
            return self.binary(args[0], ast.Sub(), args[1])
        if name in {"np.diff", "numpy.diff"} and "hidden" in result.sources:
            return replace(result, sources=result.sources | {"hidden_diff"})
        if name in {"list", "tuple", "set", "sorted", "np.array", "np.asarray"} and args:
            return args[0]
        if name == "print":
            if result.outputs and not self.uncertain:
                self.prints.append(result.outputs)
            return Fact()
        if name == "len":
            return Fact()
        return result

    def bind(self, target, value):
        if isinstance(target, ast.Name):
            self.variables[target.id] = value
        elif isinstance(target, (ast.List, ast.Tuple)):
            known = isinstance(value.value, tuple) and len(value.value) == len(target.elts)
            for i, member in enumerate(target.elts):
                self.bind(member, replace(value, value=value.value[i] if known else UNKNOWN))
        elif isinstance(target, (ast.Subscript, ast.Attribute)) and isinstance(target.value, ast.Name):
            self.variables[target.value.id] = Fact()

    def block(self, statements):
        for node in statements:
            if self.stop_line is not None and node.lineno > self.stop_line:
                break
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                result = self.value(node.value)
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    self.bind(target, result)
            elif isinstance(node, ast.AugAssign):
                result = self.binary(self.value(node.target), node.op, self.value(node.value))
                self.bind(node.target, result)
            elif isinstance(node, ast.Expr):
                self.value(node.value)
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    obj = node.value.func.value
                    if node.value.func.attr in {"append", "extend", "insert", "pop", "clear", "remove",
                                                "update", "add", "discard", "sort", "shuffle"}:
                        self.bind(obj, Fact())
            elif isinstance(node, ast.If):
                test = self.value(node.test)
                if type(test.value) is bool:
                    self.block(node.body if test.value else node.orelse)
                else:
                    self.conditional([node.body, node.orelse])
            elif isinstance(node, (ast.For, ast.While)):
                if isinstance(node, ast.For):
                    sequence = self.value(node.iter)
                    if (type(sequence.value) in (list, tuple) and not sequence.value):
                        self.block(node.orelse)
                        continue
                    self.bind(node.target, replace(sequence, value=UNKNOWN))
                else:
                    self.value(node.test)
                self.conditional([node.body, node.orelse])
            elif isinstance(node, ast.Try):
                self.conditional([node.body, *(h.body for h in node.handlers), node.orelse, node.finalbody])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.variables[node.name] = Fact()
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.expr):
                        self.value(child)

    def conditional(self, blocks):
        before = self.variables.copy()
        was_uncertain = self.uncertain
        self.uncertain = True
        changed = set()
        for block in blocks:
            self.block(block)
            changed.update(name for name, value in self.variables.items()
                           if name not in before or before[name] != value)
            self.variables.clear()
            self.variables.update(before)
        for name in changed:
            self.variables[name] = Fact()
        self.uncertain = was_uncertain


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def count_value(value):
    return value if type(value) is int and value >= 0 else None


def starting_budget(system):
    if isinstance(system, str):
        match = re.search(r"(?:BUDGET:\s*|You have\s+)(\d+)\s+forward[- ]pass units?", system, re.I)
        if match:
            return int(match[1])
    return None


def event_groups(events, rounds, initial):
    """Use the unit ledger to place paid events, and the check marker for free checks."""
    remaining = list(events)
    groups = []
    previous = initial
    for record in rounds:
        left = count_value(record.get("units_left"))
        cost = previous - left if previous is not None and left is not None else None
        group = []
        while remaining:
            event = remaining[0]
            event_cost = count_value(event.get("cost"))
            if event.get("turn") is not None:
                if event["turn"] != record["turn"]:
                    break
            elif event.get("tool") == "check":
                if "[CHECK on reserved problems" not in str(record.get("output", "")):
                    break
            elif not event_cost or cost is None or not 0 < event_cost <= cost:
                break
            group.append(remaining.pop(0))
            if cost is not None and event_cost:
                cost -= event_cost
        groups.append(group)
        previous = left
    return groups, remaining


def as_map(value):
    return value if isinstance(value, dict) else {}


def edit_map(value):
    return {name: change for name, change in as_map(value).items() if finite(change)}


def pair_rows(value):
    if not isinstance(value, list) or not value:
        return []
    return value if all(
        isinstance(row, (tuple, list)) and len(row) >= 3 and pid(row[0])
        and correctness(row[1]) is not None and correctness(row[2]) is not None for row in value
    ) else []


def pair_result(value):
    if isinstance(value, dict):
        rows = next((pair_rows(v) for k, v in value.items() if k.startswith("per_problem")), [])
        if rows or PAIR_FIELDS & value.keys():
            return value, rows
    return {}, pair_rows(value)


def cached_correctness(value):
    return {key: row["correct"] for key, row in as_map(value).items()
            if pid(key) and isinstance(row, dict) and correctness(row.get("correct")) is not None}


def set_error(call, issues):
    """Associate an observed exception with a call site when the evidence permits."""
    for _, kind, message, line in issues:
        if kind in REPLY_ISSUES or kind == "CodeRefused":
            continue
        matches = [c for c in call if c.line == line] if line is not None else []
        if "rescales a whole tensor" in message or re.search(r"u has length .* writes .* dims", message):
            matches = [c for c in matches or call if c.tool in EDITS]
        elif kind == "BudgetError":
            matches = [c for c in matches or call if c.tool in FORWARD]
        if matches:
            target = matches[-1]
            target.failed = True
            target.evidence = "报错"


def attach_events(calls, events, turn):
    """Coalesce a logged call and its static call site. Preserve extra loop executions."""
    extras = []
    for tool, count in Counter(e["tool"] for e in events).items():
        matched = [c for c in calls if c.tool == tool and not c.failed]
        records = [e for e in events if e["tool"] == tool]
        failed = [c for c in calls if c.tool == tool and c.failed]
        if len(matched) < count and len(matched) + len(failed) == count:
            matched = [c for c in calls if c.tool == tool]
        if len(matched) == count:
            targets = matched
        elif len(matched) == 1 and count > 1:
            index = calls.index(matched[0])
            targets = [matched[0], *(replace(matched[0]) for _ in range(count - 1))]
            calls[index:index + 1] = targets
        else:
            for candidate in matched:
                candidate.evidence = "候选"
            targets = [Call(tool, turn) for _ in records]
            extras.extend(targets)
        for target, event in zip(targets, records):
            target.evidence = "流水＋报错" if target.failed else "流水"
            target.result = as_map(event.get("result"))
            args = as_map(event.get("args"))
            target.ids = ids_of(event.get("ids")) or ids_of(args.get("ids")) or target.ids
    # Multiple unmatched event types must retain their observed order.
    by_tool = {tool: iter([c for c in extras if c.tool == tool]) for tool in {c.tool for c in extras}}
    for event in events:
        source = by_tool.get(event["tool"])
        extra = next(source, None) if source is not None else None
        if extra is not None:
            calls.append(extra)
    # If code sites disagree with the recorded order, prefer the recorded paid order.
    observed = [c for c in calls if c.evidence.startswith("流水")]
    order = [e["tool"] for e in events]
    if [c.tool for c in observed] != order:
        queues = {tool: iter([c for c in observed if c.tool == tool]) for tool in set(order)}
        reordered = iter(next(queues[tool]) for tool in order)
        calls[:] = [next(reordered) if c.evidence.startswith("流水") else c for c in calls]


def link_results(calls, values, prints, turn):
    candidates = [c for c in calls if c.tool == "paired_test" and not c.failed]
    pairs = [p for value in values if any(p := pair_result(value))]
    origins = [marks for marks in prints if any(m[0] == "paired_test" for m in marks)]
    logged = [c for c in candidates if c.evidence == "流水"]
    if logged:
        candidates = logged

    def record(call, result, rows):
        row_ids = tuple(row[0] for row in rows)
        if call.ids and row_ids and set(call.ids) != set(row_ids):
            return
        call.result.update(result)
        if rows:
            call.result["per_problem"] = rows
            call.ids = row_ids
        if call.evidence != "流水":
            call.evidence = "输出"

    pending = []
    associated = set()
    print_marks = origins if len(origins) == len(pairs) else [None] * len(pairs)
    for (result, rows), marks in zip(pairs, print_marks):
        row_ids = {row[0] for row in rows}
        if marks and all(t != turn for tool, t, _ in marks if tool == "paired_test"):
            continue
        sites = {line for tool, t, line in marks or () if tool == "paired_test" and t == turn}
        matching = [c for c in candidates if c.line in sites]
        if len(matching) != 1 and row_ids:
            matching = [c for c in candidates if c.ids and set(c.ids) == row_ids]
        if len(matching) == 1:
            record(matching[0], result, rows)
            associated.add(id(matching[0]))
        else:
            pending.append((result, rows))
    unmatched = [c for c in candidates if id(c) not in associated]
    if len(candidates) == 1 and (len(pending) == 1 or len(pending) == 2 and
                                 sum(bool(result) for result, _ in pending) == 1):
        for result, rows in pending:
            record(candidates[0], result, rows)
    elif len(unmatched) == len(pending):
        for call, (result, rows) in zip(unmatched, pending):
            record(call, result, rows)
    for call in candidates:
        _, rows = pair_result(call.result)
        if rows:
            call.ids = tuple(row[0] for row in rows)
            derived = {"n": len(rows), "base_acc": sum(row[1] for row in rows) / len(rows),
                       "edit_acc": sum(row[2] for row in rows) / len(rows),
                       "delta": sum(row[2] - row[1] for row in rows) / len(rows)}
            if all(len(row) >= 5 and correctness(row[3]) is not None and correctness(row[4]) is not None
                   for row in rows):
                derived.update(base_trunc=sum(row[3] for row in rows) / len(rows),
                               edit_trunc=sum(row[4] for row in rows) / len(rows))
            call.result = {**derived, **call.result}

    runs = [c for c in calls if c.tool == "run" and not c.failed]
    run_values = [v for v in values if isinstance(v, list) and v and all(
        isinstance(row, dict) and pid(row.get("id")) and correctness(row.get("correct")) is not None
        for row in v
    )]
    if len(runs) == len(run_values):
        for call, rows in zip(runs, run_values):
            row_ids = tuple(row["id"] for row in rows)
            if call.ids and set(call.ids) != set(row_ids):
                continue
            call.ids = row_ids
            call.result["rows"] = rows
            if call.evidence != "流水":
                call.evidence = "输出"


PRINTED_EDIT = re.compile(r"^set (\S+): relative change ([0-9.eE+-]+)\s*$", re.M)
PRINTED_RESET = re.compile(r"^reset (all edits|layers\.\S+)\s*$", re.M)
SCALING_ERROR = "rescales a whole tensor"


def link_edits(calls, body, before, after):
    def mark(matches, changes, evidence):
        if matches and (len(matches) == 1 or all(c.evidence != "候选" for c in matches)):
            match = matches[-1]
        else:
            match = Call("reset" if all(v is None for v in changes.values()) else "编辑",
                         calls[0].turn if calls else None)
            calls.append(match)
        match.changes.update(changes)
        match.evidence = evidence

    printed = {}
    for matrix, relative in PRINTED_EDIT.findall(body):
        try:
            relative = float(relative)
        except ValueError:
            continue
        if finite(relative):
            printed.setdefault(matrix, []).append(relative)
    for matrix, changes in printed.items():
        matches = [c for c in calls if c.tool in EDITS and c.matrix == matrix and not c.failed]
        if len(matches) == len(changes) and all(c.evidence != "候选" for c in matches):
            for call, value in zip(matches, changes):
                mark([call], {matrix: value}, "输出")
        else:
            for value in changes:
                mark(matches if len(matches) == 1 and len(changes) == 1 else [], {matrix: value}, "输出")
    for matrix, value in after.items():
        if before.get(matrix) != value:
            matches = [c for c in calls if c.tool in EDITS and c.matrix in (None, matrix) and not c.failed]
            mark(matches, {matrix: value}, "快照")
    for matrix in before.keys() - after.keys():
        mark([c for c in calls if c.tool == "reset" and c.matrix in (None, matrix) and not c.failed],
             {matrix: None}, "快照")
    for target in PRINTED_RESET.findall(body):
        for call in calls:
            if call.tool == "reset" and not call.failed and (call.matrix or "all edits") == target:
                if call.evidence != "快照":
                    call.evidence = "输出"


def usable(call):
    return not call.failed and call.evidence != "候选"


def resolved(call):
    return not call.failed and call.evidence in {"流水", "输出", "快照"}


@dataclass
class PairReview:
    call: Call
    overlap: set
    new: set
    unknown_pool: bool
    pre_wrong: str
    current_wrong: str


def only_wrong(flags):
    if any(flag is True for flag in flags):
        return "否"
    return "是" if flags and all(flag is False for flag in flags) else "不确定"


def pair_review(call, selected, incomplete, initial, last_base):
    _, rows = pair_result(call.result)
    ids = set(call.ids or ())
    flags = [call.correct if call.correct is not None else initial.get(i, last_base.get(i)) for i in ids]
    if not ids and call.correct is not None and resolved(call):
        flags = [call.correct]
    base_acc = call.result.get("base_acc")
    current = only_wrong([correctness(row[1]) for row in rows])
    if not rows and finite(base_acc) and 0 <= base_acc <= 1:
        current = "是" if base_acc == 0 else "否"
    return PairReview(call, ids & selected, ids - selected, incomplete, only_wrong(flags), current)


def negative(result):
    delta = result.get("delta")
    base, edit = result.get("base_trunc"), result.get("edit_trunc")
    criteria = []
    if finite(delta) and delta < 0:
        criteria.append(f"Δ准确率 {points(delta)}")
    if finite(base) and finite(edit) and edit > base:
        criteria.append(f"Δ截断率 {points(edit - base)}")
    return "、".join(criteria)


def apply_correctness(call, last_base, last_edit):
    _, rows = pair_result(call.result)
    for row in rows:
        last_base[row[0]] = correctness(row[1])
        last_edit[row[0]] = correctness(row[2])
    if call.tool == "run" and call.edited is not None:
        destination = last_edit if call.edited else last_base
        for row in call.result.get("rows", ()):
            destination[row["id"]] = correctness(row["correct"])


def read_failure(call, initial, last_base, last_edit):
    if call.which == "base":
        return call.correct is False or bool(call.ids and initial.get(call.ids[0]) is False)
    latest = {"last_base": last_base, "last_edited": last_edit}.get(call.which, {})
    return bool(call.ids and latest.get(call.ids[0]) is False)


def analyze(session, aggregate):
    raw_rounds = session.get("log")
    if not isinstance(raw_rounds, list) or any(not isinstance(row, dict) for row in raw_rounds):
        raise ValueError("会话缺少有效的 log 列表")
    rounds = [{**row, "turn": count_value(row.get("turn")) or index} for index, row in enumerate(raw_rounds, 1)]
    event_list = session.get("events")
    events = [e for e in event_list if isinstance(e, dict) and isinstance(e.get("tool"), str)] \
        if isinstance(event_list, list) else []
    initial_budget = starting_budget(session.get("system"))
    groups, unplaced = event_groups(events, rounds, initial_budget)
    initial = {key: correctness(flag) for key, flag in cached_correctness(session.get("base_results")).items()}
    last_base, last_edit = {}, {}
    selected, incomplete = set(), False
    previous, variables = {}, {}
    timeline, reviews, errors, edits, notes = [], [], [], [], []
    first_failure, first_edit, check_turn = None, None, None
    check = session.get("checked")
    for row, group in zip(rounds, groups):
        turn = row["turn"]
        body, feedback_check = split_feedback(str(row.get("output") or ""))
        if feedback_check is not None:
            check = check if check is not None else feedback_check
            check_turn = turn
        issues = errors_in(body)
        errors.extend((turn, kind, message) for _, kind, message, _ in issues)
        failed_lines = [line for _, kind, _, line in issues if line is not None and kind not in REPLY_ISSUES]
        scan = CodeScan(variables, turn, min(failed_lines) if failed_lines else None)
        if isinstance(row.get("code"), str) and not any(kind == "CodeRefused" for _, kind, _, _ in issues):
            try:
                tree = ast.parse(row["code"])
            except (SyntaxError, RecursionError):
                notes.append(f"第 {turn} 轮代码无法解析。")
            else:
                scan.block(tree.body)
        calls = scan.calls
        set_error(calls, issues)
        attach_events(calls, group, turn)
        if feedback_check is not None and not any(c.tool == "check" for c in calls):
            calls.append(Call("check", turn, evidence="输出", result=feedback_check))
        values = [value for _, value in printed_values(body)]
        link_results(calls, values, scan.prints, turn)
        if re.search(r"\bbase_results\b", str(row.get("code") or "")):
            for value in values:
                initial.update({i: correctness(flag) for i, flag in cached_correctness(value).items()})
        for call in calls:
            if call.tool == "check":
                check_turn = turn
                if check is None and call.result:
                    check = call.result
        current = edit_map(row["edits"]) if isinstance(row.get("edits"), dict) else previous
        link_edits(calls, body, previous, current)
        previous = current
        for call in calls:
            if call.turn is None:
                call.turn = turn
            if call.changes:
                edits.append(call)
                if first_edit is None and any(v is not None for v in call.changes.values()):
                    first_edit = call
            if not usable(call):
                if call.evidence == "候选" and call.tool in DIAGNOSIS:
                    incomplete = True
                continue
            if call.tool == "answer_text" and first_failure is None:
                if read_failure(call, initial, last_base, last_edit):
                    first_failure = call
            if call.tool == "paired_test":
                # A code site without a paid event or output gives only an inferred attempt.
                reviews.append(pair_review(call, selected, incomplete, initial, last_base))
            if call.tool in DIAGNOSIS:
                selected.update(call.ids or ())
                incomplete |= call.ids is None or call.partly_known
            apply_correctness(call, last_base, last_edit)
        timeline.extend(calls)

    timeline_tail = []
    attach_events(timeline_tail, unplaced, None)
    timeline.extend(timeline_tail)
    if check is None:
        check = next((e["result"] for e in unplaced if e["tool"] == "check" and e.get("result")), None)
    decision = session.get("decision") or aggregate.get("decision")
    if check is None:
        check = aggregate.get("check")
    if as_map(check).get("error"):
        errors.append((check_turn, "预留检查失败", str(check["error"])))
    if unplaced:
        notes.append(f"{len(unplaced)} 条流水无法定位到回合（可能是会话仍在写入，或缺少预算刻度）。")
    end_edits = edit_map(session["edits"]) if isinstance(session.get("edits"), dict) else previous
    left_values = [count_value(row.get("units_left")) for row in rounds]
    last_left = next((left for left in reversed(left_values) if left is not None), None)
    ledger = initial_budget - last_left if initial_budget is not None and last_left is not None else None
    costs = [count_value(e.get("cost")) for e in events if e["tool"] in FORWARD or "cost" in e]
    spent = sum(costs) if isinstance(event_list, list) and all(c is not None for c in costs) else None
    summary_units = count_value(aggregate.get("units_used"))
    observed_units = {n for n in (spent, ledger, summary_units) if n is not None}
    if len(observed_units) > 1:
        observed = [f"流水 {spent}", f"预算差 {ledger}", f"汇总 {summary_units}"]
        notes.append("前向单位字段不一致（" + "，".join(x for x in observed if not x.endswith("None")) + "）。")
    unit_sources = (spent, ledger, summary_units) if events else (ledger, summary_units, spent)
    units = next((n for n in unit_sources if n is not None), None)

    for tool, final in (("confirm", "submit"), ("withdraw", "withdraw")):
        candidates = [call for call in timeline if call.tool == tool and not call.failed]
        if decision == final and candidates:
            candidates[-1].evidence = "决定"
    return dict(rounds=rounds, units=units, initial_budget=initial_budget, last_left=last_left,
                timeline=timeline, reviews=reviews, errors=errors, edits=edits, end_edits=end_edits,
                first_failure=first_failure, first_edit=first_edit, check_turn=check_turn,
                check=check, decision=decision, eval=aggregate.get("eval"), notes=notes)


def points(value):
    return f"{100 * value:+.2f} 个百分点" if finite(value) else "—"


def percent(value):
    return f"{value:.2%}" if finite(value) else "—"


def rate_pair(result, left, right):
    return f"{percent(result.get(left))} → {percent(result.get(right))}"


def compact_stats(result):
    if not isinstance(result, dict) or not result:
        return "未记录"
    if result.get("error"):
        return f"失败（{result['error']}）"
    parts = []
    if result.get("n") is not None:
        parts.append(f"n={result['n']}")
    if finite(result.get("base_acc")) or finite(result.get("edit_acc")):
        parts.append("准确率 " + rate_pair(result, "base_acc", "edit_acc"))
    for field, label in (("delta", "Δ"), ("se", "SE ")):
        if finite(result.get(field)):
            parts.append(label + (f"{100 * result[field]:.2f} 个百分点" if field == "se" else points(result[field])))
    if finite(result.get("base_trunc")) or finite(result.get("edit_trunc")):
        parts.append("截断率 " + rate_pair(result, "base_trunc", "edit_trunc"))
    if finite(result.get("base_acc_finished")) or finite(result.get("edit_acc_finished")):
        parts.append("答完题准确率 " + rate_pair(result, "base_acc_finished", "edit_acc_finished"))
    if finite(result.get("wins")) and finite(result.get("losses")):
        parts.append(f"胜/负 {result['wins']}/{result['losses']}")
    if finite(result.get("p_sign_one_sided")):
        parts.append(f"单侧符号检验 p={result['p_sign_one_sided']:.3g}")
    return "，".join(parts) or "有记录，统计字段不全"


def clean(value):
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", r"\|")


def inline(value):
    return "`" + clean(value).replace("`", "′") + "`"


def table(headers, rows):
    return ["| " + " | ".join(map(clean, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(clean, row)) + " |" for row in rows)]


def turn_name(turn):
    return f"第 {turn} 轮" if turn is not None else "轮次不确定"


def first_call(timeline, tool):
    calls = [c for c in timeline if c.tool == tool]
    actual = next((c for c in calls if resolved(c)), None)
    if actual:
        return call_location(actual)
    if calls:
        if all(c.failed for c in calls):
            return "未见完成（" + call_location(calls[0]) + "）"
        return "不确定（" + call_location(calls[0]) + "）"
    return "未见"


def call_location(call):
    return f"{turn_name(call.turn)}（{call.evidence}）" if call is not None else "未见"


def call_name(call):
    name = inline(call.tool)
    if call.matrix:
        name += " " + inline(call.matrix)
    if call.ids:
        name += f" [{len(call.ids)} 题]"
    if call.line is not None:
        name += f" L{call.line}"
    return f"{name}〔{call.evidence}〕"


def relative(change):
    return f"{change:.6g}（{100 * change:.4g}%）"


def human_ids(ids, unknown="—"):
    return "、".join(inline(i) for i in sorted(ids, key=str)) if ids else unknown


def novelty(review):
    if not resolved(review.call):
        return "不确定（仅见调用代码）"
    if not review.call.ids:
        return "不确定（测试 ID 缺失）"
    if not review.new:
        return "无新增（本次 ID 均已见）"
    if review.unknown_pool:
        return "不确定（此前部分 ID 缺失）"
    return "部分新题" if review.overlap else "全为新题"


def decision_text(decision):
    return {
        "submit": "保留编辑（confirm）",
        "withdraw": "主动撤回",
        "withdraw (no decision)": "框架撤回（模型未作决定）",
        None: "尚未记录",
    }.get(decision, str(decision))


def followup(report, turn):
    if turn is None:
        return "负面输出的轮次不确定，无法判断后续顺序"
    actions = [c for c in report["timeline"] if c.turn is not None and c.turn > turn
               and (c.changes or (c.tool == "reset" and resolved(c))
                    or (c.tool in {"confirm", "withdraw"} and c.evidence == "决定"))]
    response = []
    for call in actions:
        if call.tool == "confirm":
            label = "确认保留"
        elif call.tool == "withdraw":
            label = "主动撤回"
        elif call.tool == "reset":
            label = "重置编辑"
        else:
            label = "调整编辑"
        detail = f"{turn_name(call.turn)}{label}"
        if detail not in response:
            response.append(detail)
    if report["decision"] == "withdraw (no decision)":
        response.append("框架最终撤回（模型未作决定）")
    return "，".join(response) or "未见后续已证实的调整、重置或决定"


def render_session(path, report, joined):
    calls, rounds, reviews = report["timeline"], report["rounds"], report["reviews"]
    initial, left = report["initial_budget"], report["last_left"]
    units = "不确定" if report["units"] is None else str(report["units"])
    if initial is not None and left is not None:
        units += f"（剩余 {initial} → {left}）"
    first_failure = report["first_failure"]
    failure_status = call_location(first_failure)
    if first_failure is None and any(c.tool == "answer_text" for c in calls):
        failure_status = "不确定（答错状态或执行情况不全）"
    rows = [
        ("回合数", len(rounds)),
        ("前向单位消耗", units),
        ("第一次读失败答案", failure_status),
        ("第一次读隐藏状态", first_call(calls, "hidden_states")),
        ("第一次配对测试", first_call(calls, "paired_test")),
        ("第一次成功编辑", call_location(report["first_edit"])),
        ("最终决定", f"{inline(report['decision'])}，{decision_text(report['decision'])}"
         if report["decision"] is not None else decision_text(None)),
        ("预留集检查", compact_stats(report["check"])),
        ("留出集评测", compact_stats(report["eval"]) if joined else "未合并（无匹配汇总条目）"),
    ]
    lines = [f"## {inline(path.name)}", "", *table(("指标", "记录"), rows), "",
             "### 工具次数与时间线", ""]
    counts = {}
    for call in calls:
        counter = counts.setdefault(call.tool, Counter())
        counter[call.evidence] += 1
    if counts:
        count_rows = [(inline(tool), "、".join(f"{name} {n}" for name, n in tally.items()))
                      for tool, tally in counts.items()]
        lines += table(("工具／框架动作", "次数及依据"), count_rows) + [""]
    else:
        lines += ["未见工具调用。", ""]
    by_turn = [(turn_name(row["turn"]), " → ".join(call_name(c) for c in calls if c.turn == row["turn"]) or "无")
               for row in rounds]
    unplaced = [call_name(c) for c in calls if c.turn is None]
    if unplaced:
        by_turn.append(("轮次不确定", " → ".join(unplaced)))
    if by_turn:
        lines += table(("回合", "顺序（L 为该轮代码行号）"), by_turn) + [""]

    lines += ["### 编辑与错误", ""]
    attempts = [c for c in calls if c.tool in EDITS | {"reset", "编辑"}]
    if attempts:
        edit_rows = []
        for call in attempts:
            changes = "、".join(f"{inline(k)} {relative(v) if v is not None else '已重置'}"
                                for k, v in call.changes.items())
            if not changes:
                changes = "失败" if call.failed else "未见可归因的矩阵变化"
            edit_rows.append((turn_name(call.turn), inline(call.tool), inline(call.matrix) if call.matrix else "—",
                              call.direction or "—", changes))
        lines += table(("回合", "工具", "目标", "方向来源（代码推断）", "结果"), edit_rows) + [""]
    else:
        lines += ["未见编辑或重置调用。", ""]
    end = "、".join(f"{inline(k)} {relative(v)}" for k, v in report["end_edits"].items()) or "无"
    lines += [f"- 末态矩阵快照为 {end}。快照记录会话留下的矩阵状态，是否进入留出评测由最终决定控制。"]
    weak = [c for c in attempts if c.tool == "weaken"]
    if weak:
        n = sum(any(v is not None for v in c.changes.values()) for c in weak)
        lines += [f"- 发现 `weaken` 调用痕迹 {len(weak)} 处，其中 {n} 处有输出或快照证实写入。"]
    else:
        lines += ["- 未见 `weaken` 调用。"]
    failures = [(turn, kind, msg) for turn, kind, msg in report["errors"] if kind not in REPLY_ISSUES]
    replies = [(turn, kind) for turn, kind, _ in report["errors"] if kind in REPLY_ISSUES]
    scaling = [turn_name(turn) for turn, _, msg in failures if SCALING_ERROR in msg]
    lines += ["- " + ("、".join(scaling) + "整体缩放被拒。" if scaling else "未见整体缩放被拒。")]
    if failures:
        error_counts = "、".join(f"{inline(k)} ×{count}" for k, count in Counter(k for _, k, _ in failures).items())
        lines += [f"- 可见报错 {len(failures)} 次，{error_counts}。"]
        lines += [f"  - {turn_name(turn)} {inline(kind)}，{clean(message[:240])}"
                  for turn, kind, message in failures]
    else:
        lines += ["- 可见报错 0 次。"]
    labels = {"ThinkingLimit": "思考达到 token 上限", "MissingCode": "缺少 Python 块", "OutputTruncated": "反馈文本被截断"}
    lines += [f"- {turn_name(turn)}{labels[kind]}。" for turn, kind in replies]

    lines += ["", "### 实验纪律", ""]
    if reviews:
        rows = []
        for review in reviews:
            call = review.call
            rows.append((call_location(call), human_ids(call.ids, "不确定"), human_ids(review.overlap),
                         human_ids(review.new), novelty(review), review.pre_wrong, review.current_wrong,
                         compact_stats(call.result)))
        lines += table(("配对测试", "题 ID", "与此前重叠", "新增可见 ID", "此前可见题集外",
                        "测试前已知全错", "当次底模全错", "配对结果"), rows) + [""]
        verdicts = [r.pre_wrong for r in reviews if resolved(r.call)]
        all_wrong = "否" if "否" in verdicts else (
            "是" if verdicts and verdicts == ["是"] * len(reviews)
            and not any(c.tool == "paired_test" and c.turn is None for c in calls) else "不确定"
        )
        judgment = {"是": "会话的配对测试均使用测试前已知错题。",
                    "否": "至少一次配对测试包含测试前已知答对的题。",
                    "不确定": "配对测试是否只用测试前已知错题，不确定。"}
        lines += ["- " + judgment[all_wrong]]
    else:
        lines += ["- 无可分析的配对测试，题集独立性与只测已知错题均无法判断。"]
    negatives = [(r.call.turn, "配对测试", negative(r.call.result)) for r in reviews]
    negatives.append((report["check_turn"], "预留检查", negative(as_map(report["check"]))))
    negatives = [(turn, kind, reason) for turn, kind, reason in negatives if reason]
    if negatives:
        lines += [f"- {turn_name(turn)}{kind}出现 {reason}。后续为{followup(report, turn)}。"
                  for turn, kind, reason in negatives]
    else:
        lines += ["- 未记录可判定的负面测试结果，无法评价模型怎样响应负反馈。"]
    lines += [f"- {note}" for note in report["notes"]]
    return lines


def load_json(path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层应为对象")
    return data


def main():
    parser = argparse.ArgumentParser(description="离线分析 agent8b JSON 转录，向 stdout 输出中文 Markdown。")
    parser.add_argument("json", nargs="+", type=Path, metavar="json")
    args = parser.parse_args()
    lines = [
        "# 8B 自改写会话转录分析", "",
        "- 流水与输出记录可观察的调用或结果，付费工具在生成前计费并写入流水。"
        "代码表示从 AST 推断的调用，候选表示分支、循环等执行情况未定。"
        "报错计入尝试。快照记录相对底模的 Frobenius 范数变化，终态决定以会话字段为准。",
        "- 选题比较采用严格的可见记录口径。此前用于读题、读答案、读隐藏状态或自测的题均纳入候选集。"
        "此前 ID 缺失时保留不确定性。已知错题优先依据起始诊断筛选或评分，其次用此前可见底模评分；当次配对底模评分另列。",
        "- 负面反馈按准确率下降或截断率上升标记。后续动作取更晚回合，时间顺序本身不足以确定原因。", "",
    ]
    aggregate = {}
    if RESULT_FILE.is_file():
        try:
            aggregate = load_json(RESULT_FILE)
            lines += [f"留出汇总来源为 {inline(RESULT_FILE.name)}，按正式转录文件名中的会话号关联。", ""]
        except (OSError, ValueError) as exc:
            lines += [f"留出汇总暂无法读取（{clean(exc)}）。", ""]
    else:
        lines += [f"未找到正式留出汇总 {inline(RESULT_FILE.name)}。", ""]
    status = 0
    for path in args.json:
        try:
            session = load_json(path)
            match = re.fullmatch(r"8b_session8b_(\d+)\.json", path.name)
            summary = as_map(aggregate.get(f"session{int(match[1])}")) if match else {}
            conflicts = [label for key, summary_key, label in (
                ("decision", "decision", "决定"), ("edits", "edits", "矩阵快照"), ("checked", "check", "检查结果")
            ) if key in session and summary_key in summary and session[key] != summary[summary_key]]
            if conflicts:
                summary = {}
            report = analyze(session, summary)
            if conflicts:
                report["notes"].append("转录与同编号汇总的" + "、".join(conflicts) + "不一致，跳过合并。")
            lines += render_session(path, report, bool(summary)) + [""]
        except (OSError, ValueError) as exc:
            print(f"{path}：{exc}", file=sys.stderr)
            status = 1
    print("\n".join(lines).rstrip())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
