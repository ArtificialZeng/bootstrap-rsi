"""T2 独立统计基准和 CPU 合成测试。配对输入每行对应一道题。"""

import ast
from dataclasses import dataclass
from itertools import product
from math import sqrt
from pathlib import Path
import unittest
import warnings

import numpy as np
from scipy.stats import binom, binomtest


def _binary(values):
    array = np.asarray(values)
    if array.dtype.kind not in "buif" or not np.isin(array, (0, 1)).all():
        raise ValueError("输入只能包含二元指标 0/1")
    return array.astype(np.int64)


def _paired_binary(base, edited):
    base, edited = _binary(base), _binary(edited)
    if base.shape != edited.shape or base.ndim not in (1, 2) or not base.size:
        raise ValueError("两侧配对数据必须非空、等形，且为一维或二维")
    return base, edited


def _paired_differences(base, edited):
    base, edited = _paired_binary(base, edited)
    return (edited - base).reshape(len(base), -1)


@dataclass(frozen=True)
class PairedAccuracy:
    delta: float
    se: float | None
    n_up: int
    n_down: int


def paired_accuracy(base, edited):
    differences = _paired_differences(base, edited)
    by_question = differences.mean(axis=1)
    n_questions = len(by_question)
    se = by_question.std(ddof=1) / sqrt(n_questions) if n_questions > 1 else None
    return PairedAccuracy(
        float(by_question.mean()),
        None if se is None else float(se),
        int(np.count_nonzero(differences == 1)),
        int(np.count_nonzero(differences == -1)),
    )


def sign_test(base, edited):
    """单侧精确符号检验。一维输入应为相互独立的题目配对。"""
    differences = _paired_differences(base, edited)
    if differences.shape[1] != 1:
        raise ValueError("每题多答时使用逐题符号翻转")
    wins = np.count_nonzero(differences == 1)
    losses = np.count_nonzero(differences == -1)
    return float(binom.sf(wins - 1, wins + losses, 0.5))


def bootstrap_ci(base, edited, draws=10_000, seed=0):
    """按题有放回抽样的双侧百分位区间。"""
    if draws < 1:
        raise ValueError("至少需要一次重采样")
    differences = _paired_differences(base, edited).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for start in range(0, draws, 1024):
        end = min(start + 1024, draws)
        indices = rng.integers(len(differences), size=(end - start, len(differences)))
        boot[start:end] = differences[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def sign_flip_test(base, edited, draws=100_000, seed=0):
    """按题整体翻转一次符号。draws=None 时枚举小样本的精确零分布。"""
    scores = _paired_differences(base, edited).sum(axis=1)
    scores = scores[scores != 0]
    observed = scores.sum()
    if draws is None:
        if len(scores) > 16:
            raise ValueError("精确枚举最多支持 16 道非零差值题")
        count = sum(np.dot(signs, scores) >= observed
                    for signs in product((-1, 1), repeat=len(scores)))
        return float(count / 2 ** len(scores))
    if draws < 1:
        raise ValueError("至少需要一次符号翻转抽样")
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, draws, 1024):
        size = min(1024, draws - start)
        signs = 2 * rng.integers(2, size=(size, len(scores))) - 1
        count += np.count_nonzero(signs @ scores >= observed)
    return (count + 1) / (draws + 1)


@dataclass(frozen=True)
class Decomposition:
    delta: float
    t_b: float
    t_e: float
    a_b: float | None
    a_e: float | None
    truncation_term: float | None
    finished_term: float | None


def decompose(base_correct, base_truncated, edited_correct, edited_truncated):
    """按答卷汇总。无答完样本时条件准确率未定义，零权重项仍可确定。"""
    base, edited = _paired_binary(base_correct, edited_correct)
    cut_b, cut_e = _binary(base_truncated), _binary(edited_truncated)
    if not base.shape == cut_b.shape == cut_e.shape:
        raise ValueError("正确和截断指标必须逐答卷对齐")
    if np.any(base & cut_b) or np.any(edited & cut_e):
        raise ValueError("截断答案的正确指标必须为 0")

    finished_b, finished_e = base.size - cut_b.sum(), edited.size - cut_e.sum()
    a_b = float(base.sum() / finished_b) if finished_b else None
    a_e = float(edited.sum() / finished_e) if finished_e else None
    t_b, t_e = float(cut_b.mean()), float(cut_e.mean())
    if finished_b == finished_e:
        truncation_term = 0.0
    else:
        truncation_term = None if a_b is None else (t_b - t_e) * a_b
    if not finished_e:
        finished_term = 0.0
    else:
        finished_term = None if a_b is None else (1 - t_e) * (a_e - a_b)
    return Decomposition(
        float((edited - base).mean()), t_b, t_e, a_b, a_e,
        truncation_term, finished_term,
    )


def _source_functions(filename, *names):
    """逐字提取目标函数，避免执行项目文件中的模块级 MLX 导入。"""
    source_path = Path(__file__).resolve().parents[2] / filename
    source = source_path.read_text(encoding="utf-8")
    definitions = {
        node.name: node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    namespace = {"np": np, "binomtest": binomtest}
    for name in names:
        node = definitions[name]
        function_source = ast.get_source_segment(source, node)
        line_preserving_source = "\n" * (node.lineno - 1) + function_source
        exec(compile(line_preserving_source, str(source_path), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _records(correct, truncated=None, pids=None):
    if truncated is None:
        truncated = [False] * len(correct)
    if pids is None:
        pids = range(len(correct))
    return [
        dict(pid=pid, correct=bool(ok), truncated=bool(cut), looping=False)
        for pid, ok, cut in zip(pids, correct, truncated, strict=True)
    ]


class TestStatisticsAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, paired = _source_functions("agent8b.py", "acc_f", "paired_stats")
        split, interval = _source_functions("analyze.py", "decompose", "bootstrap_ci")
        cls.project_paired = staticmethod(paired)
        cls.project_decompose = staticmethod(split)
        cls.project_bootstrap = staticmethod(interval)

    def _compare_one_answer(self, base, edited, cut_b=None, cut_e=None):
        ours = paired_accuracy(base, edited)
        base_records, edited_records = _records(base, cut_b), _records(edited, cut_e)
        project = self.project_paired(base_records, edited_records)
        self.assertAlmostEqual(ours.delta, project["delta"])
        if ours.se is None:
            self.assertTrue(np.isnan(project["se"]))
        else:
            self.assertAlmostEqual(ours.se, project["se"])
        self.assertEqual((ours.n_up, ours.n_down), (project["wins"], project["losses"]))
        self.assertAlmostEqual(sign_test(base, edited), project["p_sign_one_sided"])
        self.assertAlmostEqual(sign_flip_test(base, edited, draws=None), sign_test(base, edited))
        np.testing.assert_allclose(
            bootstrap_ci(base, edited), self.project_bootstrap(base_records, edited_records),
            atol=1e-14, rtol=0,
        )

    def test_one_answer_regular_all_correct_and_all_wrong(self):
        for base, edited in [
            ([0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1]),
            ([1, 1, 1], [1, 1, 1]),
            ([0, 0, 0], [0, 0, 0]),
        ]:
            with self.subTest(base=base, edited=edited):
                self._compare_one_answer(base, edited)
        self.assertEqual(sign_test([0, 0, 0], [1, 1, 0]), 0.25)
        self.assertEqual(sign_test([1, 1, 0], [0, 0, 0]), 1.0)
        self.assertEqual(sign_test([1, 1, 1], [1, 1, 1]), 1.0)

    def test_single_question(self):
        for base, edited in [([0], [1]), ([1], [0]), ([1], [1])]:
            with self.subTest(base=base, edited=edited):
                self._compare_one_answer(base, edited)
                value = edited[0] - base[0]
                self.assertIsNone(paired_accuracy(base, edited).se)
                self.assertEqual(bootstrap_ci(base, edited), (value, value))
        self.assertEqual(sign_test([0], [1]), 0.5)
        self.assertEqual(sign_test([1], [0]), 1.0)
        self.assertIsNone(paired_accuracy([[0, 1]], [[1, 1]]).se)

    def test_known_question_level_permutation(self):
        base = [[0, 0], [0, 0], [1, 0]]
        edited = [[1, 1], [1, 0], [0, 0]]
        self.assertEqual(sign_flip_test(base, edited, draws=None), 3 / 8)
        self.assertAlmostEqual(sign_flip_test(base, edited), 3 / 8, delta=0.006)
        self.assertEqual(sign_flip_test([1, 0], [1, 0], draws=100), 1.0)
        self.assertEqual(sign_flip_test([0] * 20, [1] * 20, draws=7), 1 / 8)

    def test_repeated_answers_change_se_ci_and_p(self):
        base = [[0, 0], [0, 0], [1, 1], [1, 1]]
        edited = [[1, 1], [1, 1], [0, 0], [0, 0]]
        pids = [0, 0, 1, 1, 2, 2, 3, 3]
        flat_base = _records(np.ravel(base), pids=pids)
        flat_edited = _records(np.ravel(edited), pids=pids)
        ours = paired_accuracy(base, edited)
        project = self.project_paired(flat_base, flat_edited)
        self.assertEqual(ours.delta, project["delta"])
        self.assertEqual(project["n"], 8)
        self.assertAlmostEqual(ours.se, 1 / sqrt(3))
        self.assertAlmostEqual(project["se"], 1 / sqrt(7))
        self.assertEqual(bootstrap_ci(base, edited), (-1.0, 1.0))
        self.assertEqual(self.project_bootstrap(flat_base, flat_edited), (-0.75, 0.75))
        self.assertEqual(sign_flip_test(base[:2], edited[:2], draws=None), 0.25)
        self.assertAlmostEqual(
            self.project_paired(flat_base[:4], flat_edited[:4])["p_sign_one_sided"], 0.0625,
        )
        with self.assertRaises(ValueError):
            sign_test(base, edited)

    def test_length_and_problem_order_mismatch(self):
        base, edited = _records([0, 1]), _records([1])
        for method in (paired_accuracy, bootstrap_ci, sign_test, sign_flip_test):
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method([0, 1], [1])
        with self.assertRaises(ValueError):
            decompose([0, 1], [0, 0], [1], [0])
        with self.assertRaises(AssertionError):
            self.project_paired(base, edited)
        self.assertEqual(self.project_bootstrap(base, edited), (1.0, 1.0))
        aligned = _records([0, 1])
        with self.assertRaises(AssertionError):
            self.project_paired(base, aligned[::-1])
        self.assertEqual(self.project_bootstrap(base, aligned), (0.0, 0.0))
        self.assertEqual(self.project_bootstrap(base, aligned[::-1]), (-1.0, 1.0))

    def test_empty_sample(self):
        for method in (paired_accuracy, bootstrap_ci, sign_test, sign_flip_test):
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method([], [])
        with self.assertRaises(ValueError):
            decompose([], [], [], [])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            paired = self.project_paired([], [])
            ci = self.project_bootstrap([], [], n=10)
        self.assertEqual(paired["n"], 0)
        self.assertTrue(np.isnan(paired["delta"]))
        self.assertTrue(all(np.isnan(bound) for bound in ci))
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def _compare_decomposition(self, base, cut_b, edited, cut_e):
        independent = decompose(base, cut_b, edited, cut_e)
        stats = self.project_paired(_records(base, cut_b), _records(edited, cut_e))
        self.assertAlmostEqual(independent.delta, stats["delta"])
        self.assertAlmostEqual(independent.t_b, stats["base_trunc"])
        self.assertAlmostEqual(independent.t_e, stats["edit_trunc"])
        for ours, existing in [(independent.a_b, stats["base_acc_finished"]),
                               (independent.a_e, stats["edit_acc_finished"])]:
            if ours is None:
                self.assertTrue(np.isnan(existing))
            else:
                self.assertAlmostEqual(ours, existing)
        return independent, self.project_decompose(stats)

    def test_identified_decomposition(self):
        for base, cut_b, edited, cut_e, expected in [
            ([0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1], (0.125, 0.125)),
            ([1, 1], [0, 0], [1, 1], [0, 0], (0.0, 0.0)),
            ([0, 0], [0, 0], [0, 0], [0, 0], (0.0, 0.0)),
            ([0], [0], [1], [0], (0.0, 1.0)),
        ]:
            with self.subTest(base=base, cut_b=cut_b, edited=edited, cut_e=cut_e):
                ours, project = self._compare_decomposition(base, cut_b, edited, cut_e)
                self.assertAlmostEqual(ours.truncation_term, expected[0])
                self.assertAlmostEqual(ours.finished_term, expected[1])
                self.assertAlmostEqual(ours.truncation_term + ours.finished_term, ours.delta)
                self.assertAlmostEqual(ours.truncation_term, project["from_less_truncation"])
                self.assertAlmostEqual(ours.finished_term, project["from_finished_accuracy"])

    def test_all_truncated_both_arms(self):
        ours, project = self._compare_decomposition([0, 0], [1, 1], [0, 0], [1, 1])
        self.assertEqual((ours.delta, ours.truncation_term, ours.finished_term), (0.0, 0.0, 0.0))
        self.assertTrue(all(np.isnan(value) for value in project.values()))
        self._compare_one_answer([0, 0], [0, 0], [1, 1], [1, 1])

    def test_no_finished_one_arm(self):
        ours, project = self._compare_decomposition([1, 0], [0, 0], [0, 0], [1, 1])
        self.assertEqual((ours.delta, ours.truncation_term, ours.finished_term), (-0.5, -0.5, 0.0))
        self.assertEqual(project["from_less_truncation"], -0.5)
        self.assertTrue(np.isnan(project["from_finished_accuracy"]))
        ours, project = self._compare_decomposition([0, 0], [1, 1], [1, 0], [0, 0])
        self.assertEqual(ours.delta, 0.5)
        self.assertEqual((ours.truncation_term, ours.finished_term), (None, None))
        self.assertTrue(all(np.isnan(value) for value in project.values()))

    def test_invalid_indicators(self):
        with self.assertRaises(ValueError):
            decompose([1], [1], [0], [0])
        with self.assertRaises(ValueError):
            decompose([0, 1], [0], [1, 0], [0, 0])
        with self.assertRaises(ValueError):
            paired_accuracy([2], [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
