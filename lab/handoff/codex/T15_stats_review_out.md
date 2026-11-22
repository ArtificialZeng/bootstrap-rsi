# T15｜Qwen3.5-4B 预算曲线统计复核

在 400 道 MATH-500 留出题上，8192 续写档的两次运行平均准确率为 **s8 52.875%、基线 47.750%**，逐题配对差为 **+5.125 个百分点**。固定这两次运行的配置，配对均值检验的点态 95% CI 为 **[+1.604, +8.646] 个百分点**，双侧 **p = 0.00444**。对本表全部 60 项比较做校正后，这项比较未达 5% 显著。8192 及中间档的估计都针对从 4096 前缀续写的协议。以下复算只读取缓存，使用 CPU，未导入环境中的实验模块。数据快照为 [q35_caps.json 第 722–898 行](../../out/q35_caps.json#L722)，SHA-256 如附录输出。

## 1. 端点复算与 JSON 对账

直接读 4096 原始 pkl 和 8192 完整续写 pkl，以 `pid` 对齐。12 份文件各有 400 个不同的 `pid`，题集一致，`ntok` 均等于 `len(out_ids)`。在原生端点，所有截断标志都与到达上限一致，截断项的 `correct` 均为假。源文件名为 `out/mech_q35-4b_eval_<臂>.pkl`，8192 添加 `.cap8192`，`s8@2000` 的文件名使用 `s8_seed2000`。准确率的分母均为 400。[复算代码见附录 1。](#附录-1复算与校正代码)

| 实验臂 | 4096 答对数 / 准确率 | 4096 截断 | 8192 答对数 / 准确率 | 8192 截断 |
|---|---:|---:|---:|---:|
| base | 99 / 24.75% | 301 | 182 / 45.50% | 218 |
| reseed | 102 / 25.50% | 298 | 200 / 50.00% | 200 |
| s8 | 103 / 25.75% | 297 | 204 / 51.00% | 196 |
| s8@2000 | 102 / 25.50% | 298 | 219 / 54.75% | 181 |
| s8rand | 87 / 21.75% | 313 | 159 / 39.75% | 241 |
| concise | 153 / 38.25% | 247 | 214 / 53.50% | 186 |

| 上限 | 合并差值（百分点） | 题级 bootstrap 95% CI（百分点） | JSON 双侧符号翻转 p | 精确符号翻转双侧 p |
|---:|---:|---:|---:|---:|
| 4096 | +0.500 | [−2.500, +3.375] | 0.80340 | 0.804452 |
| 8192 | +5.125 | [+1.625, +8.625] | 0.00485 | 0.005412 |

两端点各臂的准确率、截断率、答完但答错数及每题答对成本均与 JSON 一致。附录也复算了全部五档的臂指标和 60 项两两比较的点估计、区间及原检验 p 值，均一致。把 10,000 次抽样换成精确的经验分布卷积，4096 的上端点为 **+3.500**，比 JSON 高 **0.125 个百分点**，8192 区间不变。这是有限次 bootstrap 的分位数波动。符号翻转精确计算与 JSON 抽样值的差别同样来自有限次蒙特卡洛。[JSON 4096 第 2–179 行](../../out/q35_caps.json#L2) · [JSON 8192 第 722–898 行](../../out/q35_caps.json#L722) · [cap_table.py 第 16–38、49–74 行](../../cap_table.py#L16)

代码有几个对后续数据有影响的边界条件。配对函数使用一侧的题号索引另一侧，内部没有检查两侧题号集合完全相等，也没有在按题号建字典前拒绝重复题号。本次复算已核实这些条件。若某臂答对数为零，`max(nc, 1)` 会给出貌似有效的每题成本，其数学值应视为未定义，本次六臂均有答对。当前 `ARMS` 已列出 `s8@2000`，加入 `s8concise` 的缓存后还需更新 `ARMS` 才会将它写进 JSON。[cap_table.py 第 10–22、32–38、49–52、60–64 行](../../cap_table.py#L10)

## 2. 配对单位与论文首选统计量

每题先算两次运行的编辑减基线结果，再取平均，即 $d_i=[(Y_{i,s8}-Y_{i,base})+(Y_{i,s8@2000}-Y_{i,reseed})]/2$。在两次运行配置固定、不同题的差值可视作独立、留出题可代表目标题目分布的条件下，对 400 个 $d_i$ 以题为单位做 bootstrap，保留了同题重复的关联，适合估计点态区间。该条件推断只覆盖题目层面的变化。400 道题上的缓存准确率本身是已观察到的描述量。两次运行不足以可靠估计跨随机流或跨批次的变异，若同批的题存在额外相关性，题级区间也可能过窄。[cap_table.py 第 26–29、60–74 行](../../cap_table.py#L26)

- **题号配对仍成立。** 每个臂回答的是同一题。4096 段按原顺序分为每块 200 题，使用 1000/1001 或 2000/2001 初始化块级随机流。各臂用同一个块的初始种子有助于控制配置。但 `generate` 在一次调用开始时设种子，随后使用批量采样。完成时间和批状态会变化，因此仅凭数字相同无法断言每题共享同一随机流。本地 `mlx_lm/generate.py` 第 1418–1428、1482–1513、1826–1835 行可核查共享采样器、移除已完成序列、按活跃批大小调度。[mech.py 第 157–179、182–186、205–223 行](../../mech.py#L157) · [common.py 第 42–47 行](../../common.py#L42)
- **8192 的两批“种子”指 4096 前缀的两次运行配置。** 续写对各臂各自的截断子集重新分批，调用用 `3000 + cap + 批号` 初始化。在缓存的 8192 阶段，每个臂都用批大小 128，第 0 批都使用 11192。不同臂里同一题可能进不同批，也可能只有一侧进入续写。同批号也无法保证批成员及题内位置相同。因此不能把数据解释为同题共享随机数的受控配对，或两条从头到尾独立的种子复现。[ext.py 第 104–128 行](../../ext.py#L104) · [续写元数据示例](../../out/mech_q35-4b_eval_base.cap8192.meta.json)
- **符号翻转可作为敏感性检验。** 独立地翻转每个 $d_i$ 的符号需要联合零分布对这些翻转不变。两臂交换或差值独立且关于零对称可提供这一性质。均值为零本身不足以保证对称，实验也没有逐题随机交换编辑分配。精确枚举只去掉蒙特卡洛误差。JSON 中出现的 `p_perm=0` 表示 20,000 次模拟没有抽到同样或更极端的统计量，真实的尾概率可以大于零，且这个计数本身无法确立 $p<1/20000$。阶段小结第 47–48 行将其读为低于该阈值，需改作蒙特卡洛分辨率说明。例如 4096 的 concise 对双种子基线，精确翻转 $p=4.53225\times10^{-11}$。[cap_table.py 第 69–74 行](../../cap_table.py#L69) · [Winkler 等关于符号翻转假设的原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)
- **替代方法。** 各次运行单独用精确 McNemar 可作补充。在 8192，s8 对 base 为 67 胜 / 45 负、+5.50 点，s8@2000 对 reseed 为 53 胜 / 34 负、+4.75 点。将两组不一致对相加得到 120/79，并按 800 个独立配对计算会忽略同题重复。题目随机截距的 logistic 混合模型可描述同题相关、以固定效应纳入运行轮次；如需与本题相同的准确率差，需要再计算标准化后的边际概率差。两个初始运行配置及少量续写批难以支持稳健的种子或批方差估计。[JSON 8192 第 762–794 行](../../out/q35_caps.json#L762)

**论文首选为题级配对的准确率差及其区间。** 在上述条件下，对 $d_i$ 用均值标准误 $s_d/\sqrt{400}$ 和 t 近似检验零均值。8192 的 $d_i=-1,-0.5,0,0.5,1$ 对应 **5、51、260、66、18** 道题，SE 为 **1.7911 个百分点**，$t_{399}=2.86140$，双侧 $p=0.00443978$，95% CI **[+1.60386, +8.64614] 个百分点**。原始两个 200 题块的差值分别为 **+3.50、+6.75 点**。固定块并按题分层计算方差，敏感性结果为 $p=0.00444925$，95% CI **[+1.60304, +8.64696] 点**，仍依赖块内题级近似独立。论文可同时给出题级 bootstrap 区间，并在同一段标注这些是点态结果以及全表校正后的 p 值。[复算及分层代码见附录 1。](#附录-1复算与校正代码)

## 3. 多重比较

比较族定义为本次 JSON 中的 **5 个档位 × 12 项比较 = 60 项**。按原表原理，单次运行用精确配对符号检验，合并比较用题级精确符号翻转。Bonferroni 的原始 p 值阈值为 **0.0008333**，在 0.05 下保留 **18 项**。Holm 校正保留 **19 项**。同一轨迹重裁出的档位高度相关，两个校正方法都允许检验之间存在依赖，前提是原始 p 值在各自假设下有效。[cap_table.py 第 10–13、32–38、60–74 行](../../cap_table.py#L10) · [Holm 原始论文](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) · [复算见附录 1](#附录-1复算与校正代码)

| 通过的比较 | Bonferroni | Holm 额外通过 |
|---|---|---|
| s8 > s8rand，单次运行 | 5120、6144、7168、8192 |  |
| concise > base、reseed、s8、s8@2000、双种子 base、双种子 s8，各自比较 | 4096、5120 |  |
| concise > base、双种子 base，各自比较 | 6144 |  |
| concise > reseed |  | 6144（调整 p = 0.04534） |

8192 双种子 **s8 对基线** 的精确翻转 p 经 Bonferroni / Holm 调整为 **0.32475 / 0.20567**。随机符号对照低于基线的差异也未通过全表校正。表中 s8 高于随机符号对照的结论只覆盖本实验抽到的这一个符号配置。[mech.py 第 210–216 行](../../mech.py#L210)

统一用题级配对 t 检验重做 60 项，Bonferroni / Holm 分别保留 **19 / 20** 项。在前表基础上，Bonferroni 多保留 6144 的 concise > reseed（调整 p = 0.04574），Holm 多保留 8192 的 concise > base 单次运行（调整 p = 0.04768）。8192 双种子 s8 对基线的 t 检验调整 p 为 **0.26639 / 0.16871**。如果研究方案事先把比较族限定为双种子 s8 对基线的五个档位，8192 的 Bonferroni 调整 p 为配对 t 的 **0.02220** 或精确符号翻转的 **0.02706**，其余四档均未通过。这个较窄的结果要求独立于结果确定比较族。预算是否改变编辑效应，需要直接检验跨档的效应差。[复算见附录 1。](#附录-1复算与校正代码)

## 4. 离线重裁与续写接缝

五档数据中，4096 使用原始生成结果。5120、6144、7168 对 8192 的缓存轨迹按 `ntok < cap` 保留已完成答案，其余算截断；8192 本身也使用经过续写的轨迹。对这些实际存储的轨迹，重裁准确率是按这个规则准确计数的结果。逐文件核对发现，4096 已答完记录完整保留，所有续写记录的 prompt 和前 4096 输出 token 与源文件一致，六个续写元数据中的源 SHA-256 也都匹配。代码是在原 prompt 拼接原输出 token 后重预填，沿用本臂权重并另设批种子。[cap_table.py 第 16–22 行](../../cap_table.py#L16) · [ext.py 第 104–140、145–150 行](../../ext.py#L104) · [复算见附录 2](#附录-2接缝核对代码)

| 实验臂 | 在 4096 续写 | 重裁至 5120 / 6144 / 7168 / 8192 时，较原 4096 新增答对数 |
|---|---:|---:|
| base | 301（75.25%） | 23 / 47 / 69 / 83 |
| reseed | 298（74.50%） | 24 / 57 / 78 / 98 |
| s8 | 297（74.25%） | 35 / 62 / 84 / 101 |
| s8@2000 | 298（74.50%） | 33 / 72 / 91 / 117 |
| s8rand | 313（78.25%） | 12 / 35 / 58 / 72 |
| concise | 247（61.75%） | 18 / 35 / 44 / 61 |

各臂共用同一续写规则，接缝暴露率与批构成因臂而异。s8/base 有 **258** 题两侧都续写，其中 **12** 题续写批号不同，另有 **39/43** 题仅一侧续写。s8@2000/reseed 对应 **263、3、35/35**。concise/base 对应 **233、64、14/68**，s8rand/base 对应 **277、6、36/24**。即使每条受影响轨迹存在同等大小的平均扰动，各臂受影响比例不同也足以使总准确率的扰动无法抵消。这些计数衡量接缝暴露，无法把实际观察到的臂差异归因于接缝。[复算见附录 2。](#附录-2接缝核对代码)

在同一个精确前缀上，如果重预填和持续解码给出同样的模型状态与条件采样分布，使用新的独立随机抽样会改变实现出的回答，但不会改变边际准确率。现实中重预填、缓存路径和动态分批可能改变数值或采样，单次连续生成的分布等价性在现有数据上**不确定**。已存的探针只测了 6 条 base 单序列、各重放 700 个输出 token，报告 argmax 同为 6/6、top-20 集合相同 4/6、温度 0.6 的 softmax KL 最大约 $1.89\times10^{-6}$。这项测量没有覆盖真实 4096 接缝、多臂动态批或最终正确率，也没有包含 top-k/top-p 过滤后的分布。同一个数值种子下，用更大上限从头运行可能改变批调度，其他题的早期轨迹也可能随之变化。因而现有档位的系统偏差方向、大小和是否对称均不确定。[check_prefill.py 第 17–41 行](../../check_prefill.py#L17) · [探针输出第 1–62 行](../../out/check_prefill_q35-4b.json#L1) · [common.py 第 42–47 行](../../common.py#L42)

## 5. 答完即正确的复核

8192 六臂共 **1178** 个答完记录。缓存中 `correct=True` 的计数为 **182/182、200/200、204/204、219/219、159/159、214/214**，且每条都能从最后一个 `</think>` 之后抽出成对花括号的最终 `\boxed{}`。按实验臂及是否续写分层，固定抽样种子 `261121885`，抽取以下 15 条与本地 MATH-500 的标准答案人工比对，**15/15 等价**。答完即正确的 100% 是上述缓存得到的观察值，人工抽查逐条核验了附表中的 15 条。这些存档的分数变化都来自完成状态变化。完成状态受实验臂影响，各臂答完的题目集合可不同，因此这个计数无法单独识别编辑是否也改变了解题能力。[common.py 第 11–26、59–89 行](../../common.py#L11) · [JSON 8192 第 722–759 行](../../out/q35_caps.json#L722) · [抽样代码见附录 3](#附录-3抽样和最终答案提取)

样本题号均带前缀 `test/`、后缀 `.json`。行号来自本机缓存 `datasets--HuggingFaceH4--MATH-500/snapshots/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl`，SHA-256 为 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`。

| 臂 | 题号 | 标准答案行 | 输出 token | 最后一个框 | 标准答案 |
|---|---|---:|---:|---|---|
| base | algebra/776 | 418 | 2130 | `-2` | `-2` |
| base | number_theory/515 | 7 | 5176 | `27` | `27` |
| base | prealgebra/1436 | 217 | 6625 | `6 \text{ Trinkets}` | `6`（题目以 Trinkets 为单位） |
| reseed | algebra/841 | 173 | 2855 | `350` | `350` |
| reseed | intermediate_algebra/1555 | 204 | 4288 | `12` | `12` |
| s8 | number_theory/627 | 36 | 3564 | `3` | `3` |
| s8 | algebra/2430 | 447 | 5643 | `8` | `8` |
| s8 | algebra/694 | 259 | 4754 | `16` | `16` |
| s8@2000 | number_theory/572 | 4 | 1429 | `9` | `9` |
| s8@2000 | intermediate_algebra/690 | 365 | 7422 | `(1, -16, -4, 43)` | `(1,-16,-4,43)` |
| s8rand | prealgebra/1238 | 343 | 2195 | `2` | `2` |
| s8rand | prealgebra/1233 | 68 | 6224 | `10\%` | `10`（题目问百分数） |
| concise | algebra/509 | 332 | 3115 | `\dfrac{13}{6}` | `\frac{13}{6}` |
| concise | number_theory/1287 | 281 | 4998 | `8` | `8` |
| concise | intermediate_algebra/199 | 440 | 4701 | `x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1` | 同左 |

## 附录 1｜复算与校正代码

在 `exp/` 下由 `../.venv/bin/python -B -` 执行。附录 2、3 可接在附录 1 后运行。

```python
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_1samp

out = Path("out")
summary_path = out / "q35_caps.json"
report = json.loads(summary_path.read_text())
caps = (4096, 5120, 6144, 7168, 8192)
arms = ("base", "reseed", "s8", "s8@2000", "s8rand", "concise")
paths, data = {}, {}
for endpoint in (4096, 8192):
    for arm in arms:
        stem = "s8_seed2000" if arm == "s8@2000" else arm
        suffix = "" if endpoint == 4096 else ".cap8192"
        path = out / f"mech_q35-4b_eval_{stem}{suffix}.pkl"
        with path.open("rb") as f:
            rows = pickle.load(f)
        indexed = {r["pid"]: r for r in rows}
        assert len(indexed) == len(rows) == 400
        assert all(r["ntok"] == len(r["out_ids"]) <= endpoint for r in rows)
        assert all(r["truncated"] == (r["ntok"] == endpoint) for r in rows)
        assert not any(r["truncated"] and r["correct"] for r in rows)
        paths[endpoint, arm], data[endpoint, arm] = path, indexed
ids = sorted(data[4096, "base"])
assert all(set(rows) == set(ids) for rows in data.values())

def p_flip(d):
    weights = [abs(int(2 * x)) for x in d if x]
    counts = [1]
    for weight in weights:
        updated = [0] * (len(counts) + weight)
        for s, count in enumerate(counts):
            updated[s] += count
            updated[s + weight] += count
        counts = updated
    total, observed = sum(weights), abs(int(2 * d.sum()))
    return sum(count for s, count in enumerate(counts)
               if abs(2 * s - total) >= observed) / (2 ** len(weights))

def exact_boot_ci(d):
    # Convolve the empirical PMF on {-1, -.5, 0, .5, 1} for a sum of n draws.
    n = len(d)
    one_draw = np.bincount((2 * d + 2).astype(int), minlength=5) / n
    dist = np.array([1.0])
    for _ in range(n):
        dist = np.convolve(dist, one_draw)
    return (np.searchsorted(np.cumsum(dist), (0.025, 0.975)) - 2 * n) / (2 * n)

all_tests = []
pooled = "s8 (2 seeds) - base (2 seeds)"
for cap in caps:
    endpoint = 4096 if cap == 4096 else 8192
    y, wrong = {}, {}
    for arm in arms:
        rows = [data[endpoint, arm][pid] for pid in ids]
        if cap == endpoint:
            finished = np.array([not r["truncated"] for r in rows])
        else:
            finished = np.array([r["ntok"] < cap for r in rows])
        stored_correct = np.array([r["correct"] for r in rows])
        y[arm] = (finished & stored_correct).astype(int)
        wrong[arm] = finished & ~stored_correct
        correct = int(y[arm].sum())
        values = dict(acc=correct / len(ids), trunc=1 - finished.mean(),
                      fin_wrong=wrong[arm].sum(),
                      tok_per_correct=sum(min(r["ntok"], cap) for r in rows) / correct)
        expected = report[str(cap)]["arms"][arm]
        assert all(np.isclose(values[k], expected[k], rtol=0, atol=1e-12) for k in values)
        if cap == endpoint:
            print(cap, arm, correct, len(ids) - finished.sum(), values["fin_wrong"])

    y["base (2 seeds)"] = (y["base"] + y["reseed"]) / 2
    y["s8 (2 seeds)"] = (y["s8"] + y["s8@2000"]) / 2
    for name, item in report[str(cap)]["pairs"].items():
        a, b = name.split(" - ")
        d = y[a] - y[b]
        assert d.mean() == item["delta"]
        draws = np.random.default_rng(0).choice(d, size=(10000, len(ids)), replace=True)
        ci = np.quantile(draws.mean(axis=1), (0.025, 0.975))
        assert np.allclose(ci, item["ci"], rtol=0, atol=1e-12)
        p = p_flip(d)
        w, l = int((d > 0).sum()), int((d < 0).sum())
        if "p" in item:
            assert (w, l) == (item["wins"], item["losses"])
            assert np.isclose(p, item["p"], rtol=0, atol=1e-14)
            assert (wrong[b] & (y[a] == 1)).sum() == item["wc"]
            assert (wrong[a] & (y[b] == 1)).sum() == item["cw"]
        else:
            assert (w, l) == (item["pos"], item["neg"])
            nz = d[d != 0]
            rng = np.random.default_rng(1)
            flip = (rng.choice([-1, 1], (20000, len(nz))) * nz).sum(axis=1)
            assert (np.abs(flip) >= abs(nz.sum())).mean() == item["p_perm"]
        test = ttest_1samp(d, 0)
        all_tests.append((name, cap, p, float(test.pvalue)))
        if name == pooled and cap == endpoint:
            print(cap, "pooled", d.mean(), ci, "exact boot", exact_boot_ci(d),
                  "flip", p, "t", test.statistic, "p", test.pvalue,
                  "CI", test.confidence_interval(), "counts", np.unique(d, return_counts=True)[1])

assert len(all_tests) == 60
print("JSON SHA256", hashlib.sha256(summary_path.read_bytes()).hexdigest())
for p_column in (2, 3):
    ranked = sorted(all_tests, key=lambda row: row[p_column])
    running = 0.0
    for i, row in enumerate(ranked):
        p = row[p_column]
        bonf = min(1, len(ranked) * p)
        running = max(running, (len(ranked) - i) * p)
        holm = min(1, running)
        if holm <= 0.05 or row[:2] == (pooled, 8192):
            print(p_column, row[:2], "raw", p, "Bonferroni", bonf, "Holm", holm)

original_order = list(data[4096, "base"])
def correct_at_8192(arm):
    return np.array([int(data[8192, arm][pid]["correct"]) for pid in original_order])

d = (correct_at_8192("s8") - correct_at_8192("base")
     + correct_at_8192("s8@2000") - correct_at_8192("reseed")) / 2
x = d.reshape(2, 200)
v = x.var(axis=1, ddof=1) / 200 / 4
se = v.sum() ** 0.5
df = v.sum() ** 2 / (v ** 2 / 199).sum()
print("block means", x.mean(axis=1), "p", 2 * t.sf(abs(d.mean() / se), df),
      "CI", d.mean() + np.array([-1, 1]) * se * t.ppf(0.975, df))
```

关键输出如下，区间和差值为比例单位。

```text
4096 pooled 0.005 [-0.025  0.03375] exact boot [-0.025  0.035]
     exact flip 0.8044520418268285
8192 pooled 0.05125 [0.01625 0.08625] exact boot [0.01625 0.08625]
     exact flip 0.0054124341312294716
     t 2.8613961171267683 p 0.004439779923434166
     CI 0.01603859626848319 0.08646140373151681 counts [5 51 260 66 18]
block means [0.035  0.0675] p 0.004449245926699982 CI [0.01603042 0.08646958]
JSON SHA256 f6b6f8ea0a3845bb869b547bf1438c3cb1c4d54ef1857d2db1b8e9c54e3710ec
```

## 附录 2｜接缝核对代码

```python
batches = {}
for arm in arms:
    before, after = data[4096, arm], data[8192, arm]
    assert list(before) == list(after) == original_order
    source = paths[4096, arm]
    meta = json.loads(paths[8192, arm].with_suffix(".meta.json").read_text())
    assert meta["src_sha"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert meta["to"] == 8192 and meta["bs"] == 128 and meta["limit"] is None
    joined = []
    for pid, a in before.items():
        b = after[pid]
        assert a["prompt_ids"] == b["prompt_ids"]
        assert a["out_ids"] == b["out_ids"][:a["ntok"]]
        if a["truncated"]:
            joined.append(pid)
        else:
            assert a == b
    batches[arm] = {pid: j // meta["bs"] for j, pid in enumerate(joined)}
    extra = [sum(r["correct"] and r["ntok"] < cap for r in after.values())
             - sum(r["correct"] for r in before.values()) for cap in caps[1:]]
    print(arm, "continued", len(joined), "extra correct", extra)

for a, b in (("s8", "base"), ("s8@2000", "reseed"), ("concise", "base"), ("s8rand", "base")):
    u, v = batches[a], batches[b]
    both = u.keys() & v.keys()
    print(a, b, len(both), sum(u[pid] != v[pid] for pid in both),
          len(u.keys() - v.keys()), len(v.keys() - u.keys()))
```

## 附录 3｜抽样和最终答案提取

```python
import random
import re

gold_path = (Path.home() / ".cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500"
             / "snapshots/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl")
with gold_path.open() as f:
    gold = {r["unique_id"]: (line, r["answer"])
            for line, r in enumerate(map(json.loads, f), 1)}
print("gold SHA256", hashlib.sha256(gold_path.read_bytes()).hexdigest())
assert set(ids) <= gold.keys()

def final_box(text):
    tail = text.rsplit("</think>", 1)[-1]
    for match in reversed(list(re.finditer(r"\\boxed\s*\{", tail))):
        start = match.end()
        depth = 1
        for pos in range(start, len(tail)):
            ch = tail[pos]
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return tail[start:pos]
    return None

rng = random.Random(261121885)
for arm in arms:
    before, after = data[4096, arm], data[8192, arm]
    finished = [r for r in after.values() if not r["truncated"]]
    assert all(r["correct"] and final_box(r["text"]) is not None for r in finished)
    print(arm, "finished", len(finished))
    for continued in (False, True):
        candidates = sorted((r for r in finished if bool(before[r["pid"]]["truncated"]) == continued),
                            key=lambda r: r["pid"])
        count = 1 + (continued and arm in ("base", "s8", "concise"))
        for r in rng.sample(candidates, count):
            pid = r["pid"]
            print(arm, pid, *gold[pid], r["ntok"], final_box(r["text"]))
```

