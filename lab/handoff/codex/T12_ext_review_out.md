# T12 · 多预算续写审查

`ext.py` 给出的是**固定已观测前缀的分段条件续写**。在同一 checkpoint、编辑、tokenizer 和停止规则下，只要重新预填能恢复同一条件采样分布，用新随机数抽后缀就成立。现有代码保存了原始 token ID，也在重预填前应用了该臂编辑。重新预填与原逐 token 解码的数值等价性不确定，8192、16384 档应标为续写结果。论文第二种协议的 490 题评测是每题 8 个答案、上限 16,384 token。本项目每题 1 个答案。 [ext.py:78–109](../../ext.py#L78) · 协议规格:199–202 · 协议规格:216–218

### 高 · 1. 同分布需要验证，题目配对可以保留

设前缀为 `x`、后缀为 `y`。在相同条件分布下，保存的 `x` 由原模型生成，新的独立随机数仍可按 `P_arm(y | x)` 抽样，联合分布由 `P_arm(x) P_arm(y | x)` 给出。只有未触发停止的前缀需要继续。种子变化本身既不会使前缀失效，也无法补偿 logits 或采样候选集合变化。现有 `common.generate` 每次只设置一次全局 RNG。4K 配置为 128 条 completion，续写默认 64/32，每档根据**各臂自己的截断题集合**重新编号和分批，批内活跃序列还会随终止而变化。同一题在多个臂中即使落到同号批次，也可能取到不同随机数。`reseed` 在 4K 用 2000 种子家族，续写阶段使用与其他臂相同的按批号公式。即使直接从原提示重跑并沿用原全局 seed 和 batch，改变上限也可能改变后续题目的入批时机和前缀。 [common.py:42–56](../../common.py#L42) · [mech.py:156–178](../../mech.py#L156) · [mech.py:215–218](../../mech.py#L215) · [4K 配置:7](../../out/mech_q35-4b_config.json#L7) · [ext.py:62–97](../../ext.py#L62) · [MLX-LM 批内采样](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L1384-L1444) · [MLX-LM 动态调度](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L1812-L1864)

**最小修补。** 给续写缓存标记生成模式和旧种子规则，统计表按题号配对。需要新采样协议时，为每题在**绝对输出位置**分配显式 RNG key，把现有温度、top-p 和 top-k 的顺序保留，并在 categorical 调用处传入 key。普通对照臂共用 `seed_family`，`reseed` 单独取值。这个改法对后缀生效，历史 4K 全局 RNG 已经发生的抽样无法重配。 [MLX-LM 采样器](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/sample_utils.py#L11-L77) · [MLX-LM categorical](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/sample_utils.py#L269-L273) · [MLX 随机 key](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/random.cpp#L27-L34)

```python
# 当前产物口径，写进续写元数据。
metadata.update(mode="conditional_prefill", rng_scheme="3000+target_cap+batch_index")

# 新一轮生成可在现有 sampler 的抽样处使用显式 key。
payload = json.dumps([seed_family, pid, absolute_output_pos]).encode("utf-8")
seed = int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little")
token = mx.random.categorical(filtered_logprobs / 0.6, key=mx.random.key(seed))
```

### 高 · 2. 重预填经过正确的掩码路径，位级等价仍不确定

按当前项目记录的 MLX-LM 0.32.2 核对固定版本上游源码。这里的 `batch_generate` 原生预填会把不等长输入**右填充**，全注意力的 `BatchKVCache.finalize` 才把填充移到左侧并修正 offset，后续因果掩码屏蔽左填充。Qwen3.5 `forward` 分别调用 `create_attention_mask` 和 `create_ssm_mask`。后者从线性层 `ArraysCache` 取掩码，卷积路径将填充 QKV 置零并按有效长度保留真实末态，Gated DeltaNet 跳过被掩码的递推步。静态路径支持正确处理填充。`common.py` 中读隐藏态的手写掩码并未参与 `ext.py` 的生成。 [项目版本记录:5](T8_throughput_out.md#L5) · [MLX-LM 分块预填](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L1209-L1238) · [KV 的 padding 和 offset](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/cache.py#L938-L980) · [ArraysCache 掩码](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/cache.py#L674-L706) · [create_ssm_mask](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/base.py#L58-L61) · [Qwen3.5 卷积和掩码分派](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_5.py#L150-L198) · [Qwen3.5 模型前向](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_5.py#L280-L296) · [common.py:142–158](../../common.py#L142)

原回答逐 token 走 decode，续写时整段是多 token prefill。不同张量形状、掩码和 Gated DeltaNet kernel 路径可能改变浮点舍入及 top-k/top-p 边界，方向与大小不确定。将 `--bs` 改回 128 也无法恢复原调度和计算形状。**最小修补**是保留当前续写并明确标记。需要可审计的一次生成做法时，从原 prompt 一次生成到 16384，再用该批轨迹复切小上限。需要精确接续历史轨迹时，则要在原生成中保存每题全注意力 KV、卷积与递推状态、下一 token 或 logits、停止状态及采样 RNG 或调度进度。仅持久化 prompt cache 会遗漏 MLX 已提前抽取的下一 token。以后验证重预填时，可先对同一串已存 token 做逐 token 教师强制回放，再比较分块预填在接点的 logits、采样候选集合及末态，单条和不等长原生批分别校验。 [MLX-LM Gated DeltaNet 掩码与更新](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/gated_delta.py#L88-L147) · [MLX-LM kernel 分派](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/gated_delta.py#L429-L491) · [MLX-LM 下一步预抽样及 cache](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L1418-L1446)

```python
# 新一轮一次生成后复切。所有小档来自这批新轨迹。
_, full_ids = generate(model, tok, original_prompt_ids, 16384, seed=seed, bs=bs)
for cap in (4096, 8192, 16384):
    for ids, answer in zip(full_ids, gold_answers, strict=True):
        kept = list(ids[:cap])
        score = grade(tok.decode(kept), len(kept), cap, answer)
```

### 高 · 3. 正式缓存可静默混入过期或不完整的轨迹

`.part{n}` 只记录行索引。改 `--bs`、更换源文件或改变行序时，旧 part 会被直接 `update` 进新结果。生成行数少于预期时，`zip` 会吞掉缺行，`new.get(i, r)` 又会把旧上限的截断记录送进新档。分片和正式 pickle 都直接覆盖目标，进程中断可能留下半个文件，正式目标此后只凭存在就被视为完成。16384 档还会把 8192 正式文件当源，继承其中的污染。 [ext.py:78–114](../../ext.py#L78)

**最小修补。** 同一个运行指纹涵盖源文件 SHA-256、目标上限、`bs`、`limit`、实际编辑和 checkpoint、采样规则与后端版本。划分批次后才检查分片，逐条核对 PID、原 prompt、旧输出前缀和计分标记。正式 pickle 保持列表格式，可用旁车清单保存运行指纹和 pickle 的 SHA-256。在同一路径加排他锁，并把旁车作为最后发布的完成标记。后续档和续跑都应校验旁车，没有可验证来源的旧文件需单独保留。 [ext.py:89–114](../../ext.py#L89) · [analyze.py:79–96](../../analyze.py#L79)

```python
idx = todo[bi:bi + bs]
if os.path.exists(pp):
    part_meta, part = read_pickle(pp)
    if part_meta != run_meta or set(part) != set(idx):
        raise ValueError(f"分片配置或题号不一致：{pp}")
    for i in idx:
        check_extension(R[i], part[i], a.to)  # 验 PID、prompt、前缀、长度和标记
else:
    part = {i: make_record(R[i], extra) for i, extra in zip(idx, ids, strict=True)}
    atomic_pickle((run_meta, part), pp)
new.update(part)
if set(new) != set(todo):  # 放在所有分片处理完以后
    raise ValueError("续写题目有缺失或重复归属")

def atomic_pickle(value, path):
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            pickle.dump(value, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
```

### 高 · 4. 续写的 checkpoint 和编辑方向缺少来源绑定

`s8` 的默认 `+0.5`、`s8rand` 的逐层随机符号和 `concise` 复用保存的 prompt，与 `mech.py` 静态一致。两处 `glob(...)[0]` 各自在启动时挑快照，不能从历史缓存确认选中的 revision 相同。另有一个实际可触发的通用错误。`edit` / `random` 的原运行可以使用 `mech.py` 内生成的方向缓存，但 `ext.py` 在省略 `--dirs` 时会对 `None` 取下标。手填不同方向文件还可能使 `edit` 的后缀来自另一套权重。`random` 的确定性方向只取原方向的维度，但仍需绑定层、维度和 α。 [ext.py:26–49](../../ext.py#L26) · [ext.py:65–72](../../ext.py#L65) · [mech.py:79–99](../../mech.py#L79) · [mech.py:107–131](../../mech.py#L107) · [mech.py:181–218](../../mech.py#L181)

**最小修补。** 初始生成时将实际模型、tokenizer revision 和方向文件摘要写进元数据。续写优先使用源运行的配置，并在加载模型前验证；历史缓存上的相同 revision 如无额外记录，仍属不确定。 [mech.py:87–99](../../mech.py#L87) · [ext.py:65–72](../../ext.py#L65)

```python
snapshot = Path(HF) / MODELS[a.model] / "snapshots" / source_meta["revision"]
if not snapshot.is_dir():
    ap.error("原运行的模型快照不可用")
if kind == "edit" or kind.startswith("random"):
    dirs_path = Path(a.dirs or source_config.get("dirs") or f"{OUT}/{tag}_dirs.pkl")
    if not dirs_path.is_file() or sha256_file(dirs_path) != source_meta["dirs_sha256"]:
        ap.error("方向文件与原运行不一致")
    dirs = read_directions(dirs_path)  # .npz 用 np.load，原内部缓存用 pickle
model, tok = load(str(snapshot))
```

### 中 · 5. 整段解码路径正确，评分边界需明确

固定版本 MLX-LM 的 `batch_generate` 本身以 `tokenizer.decode(results[uid])` 生成 `texts`，并返回同一组 ID。相同 tokenizer 和相同 ID 序列下，`ext.py` 的整段 `tok.decode(out_ids)` 与原路径一致，且允许在拼接处正确解码跨 token 的字节序列。`mech.py` 的 base-only `_cont` 先分别解码再拼字符串，存在拼接处字节或空格归属差异，所以它的评分结果不能当作 `ext.py` 的一致性校验。续写新增 `</think>` 可改变评分所在的区段，新增或补完的最后一个平衡 `\boxed{}` 也会按原规则改变评分。未出现 `</think>` 时，现有 `grade` 会在全文中查找。EOS 与长度上限同时命中时，该版本先判长、再以 stop 优先，并从返回 ID 剔除 EOS。因此 `ntok` 是**返回的输出 ID 数**，最后一步为 EOS 时可完成且 `ntok < cap`。 [ext.py:101–109](../../ext.py#L101) · [common.py:59–89](../../common.py#L59) · [mech.py:239–249](../../mech.py#L239) · [MLX-LM 停止判定](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L1480-L1501) · [MLX-LM EOS 和返回文本](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py#L1977-L2017)

**最小修补。** 保留整段解码，首次使用旧缓存前抽检或逐条验证原文本。值得固定的评分样例包括续写补全半个框、续写新增第二个框、`</think>` 前后各有框和 EOS 正好命中上限。 [ext.py:96–109](../../ext.py#L96) · [common.py:77–89](../../common.py#L77)

```python
prefix_ids = list(r["out_ids"])
if tok.decode(prefix_ids) != r["text"]:
    raise ValueError(f"旧文本与当前 tokenizer 不一致：{r['pid']}")
if r["truncated"] and set(prefix_ids).intersection(tok.eos_token_ids):
    raise ValueError(f"截断前缀含终止 token：{r['pid']}")
out_ids = prefix_ids + list(extra)
text = tok.decode(out_ids)
g = grade(text, len(out_ids), a.to, gold[r["pid"]])
```

### 中 · 6. 正数 smoke 文件与正式文件分开，路径判定仍有漏洞

正整数 `--limit` 最后写 `.test.pkl` 和 `.test.part*.pkl`，正式扩档只读取正式上档，所以常规 smoke 不会进入正式缓存。已有正式目标时，代码在选定测试路径前就 `continue`。`--limit 0` 会触发正式全量分支，负值按 Python 切片解释，不同正 limit 共享同一测试分片。现有 smoke 的最终列表还混有未续写的旧截断行，用它直接算高档准确率会把旧截断当高档截断。`assert` 对上限只做整倍数约束，`-O` 会跳过检查。 [ext.py:53–92](../../ext.py#L53) · [ext.py:113–116](../../ext.py#L113)

**最小修补。** 先选定测试命名空间再检查缓存，smoke 明确只输出处理过的样本，其汇总以该子集为分母。生产路径读取正式上档。 [ext.py:79–92](../../ext.py#L79)

```python
prev = {8192: 4096, 16384: 8192}.get(a.to)
if prev is None:
    ap.error("--to 应为 8192 或 16384")
if a.limit is not None and a.limit <= 0:
    ap.error("--limit 应为正整数")
if a.limit is not None:
    dst = dst.removesuffix(".pkl") + f".test{a.limit}.pkl"
# 定好 dst 后才校验其最终文件，并用 dst 派生分片命名。
todo = todo[:a.limit] if a.limit is not None else todo
out = [new[i] for i in todo] if a.limit is not None else [new.get(i, r) for i, r in enumerate(R)]
```

### 中 · 7. 记录字段兼容，分析入口需按题对齐并逐档取实测文件

正常续写用 `len(prefix_ids) + len(extra)` 评出 `ntok`、`truncated` 和 `correct`，原先已完成的回答低于新上限，可以原样保留。这些字段兼容 `analyze.tokens`、`analyze.transitions` 和 `analyze.bootstrap_ci`。其中后两个函数内部按位置 `zip`，传入记录前要拒绝空样本、重复 PID 和两侧 PID 集合不等，并重排到同一顺序。`budget_curve` 固定 1024–4096，可在真实采样预算内复切存档的轨迹。若把它的 cap 列表直接改长，旧截断记录会被误判为在新上限前完成。逐档统计应使用对应正式文件的标记。 [ext.py:101–114](../../ext.py#L101) · [common.py:77–89](../../common.py#L77) · [analyze.py:24–62](../../analyze.py#L24)

**最小修补。** 源和目标分别做以下校验，另在写正式结果前检查所有应续写题均已更新。 [ext.py:83–114](../../ext.py#L83)

```python
def check_record(r, cap):
    n = r["ntok"]
    if n != len(r["out_ids"]) or not 0 <= n <= cap:
        raise ValueError(f"{r['pid']} 的输出长度无效")
    if r["truncated"] != (n == cap) or (r["truncated"] and r["correct"]):
        raise ValueError(f"{r['pid']} 的上限或计分标记无效")

for r in R:
    check_record(r, prev)
for r in out:
    check_record(r, a.to)
```

### 4096 档只读核对

以已经存在的三个 4096 正式 pickle 做 CPU 核对。每臂有 400 个唯一 PID，题集相同，1200 条记录均满足 ID 长度和计分标记约束。三臂所有完成回答都正确，因此这些配对准确率差完全来自 `T↔C`。 base · s8 · s8rand

| 实验臂 | 答对 / 400 | 截断 / 400 | 准确率 | 截断率 |
| --- | ---: | ---: | ---: | ---: |
| base | 99 | 301 | 24.75% | 75.25% |
| s8 | 103 | 297 | 25.75% | 74.25% |
| s8rand | 87 | 313 | 21.75% | 78.25% |

`s8` 对 `s8rand` 为 **+4.00 个百分点**，47 胜 / 31 负，单侧 p = 0.04439，双侧 p = 0.08878。本次检查时，`mech_q35-4b_summary.json` 尚未出现。现有 `mech.py` 在全部实验臂跑完后才保存该映射。`queue5.sh` 两个扩展调用都没有请求 `reseed`，因此它的高预算档可为空。高档续写实测及重预填等价性仍未验证。 [mech.py:223–237](../../mech.py#L223) · queue5.sh:10–15

## 分析脚本用法

在 `exp/` 下执行。第一条使用临时目录和合成记录；第二条读取正式清单并在 `handoff/codex/` 生成 `analyze_caps_q35-4b.md` 和 `analyze_caps_q35-4b.json`。可用 `--output-dir` 指定其他报告目录。脚本只读取受信任的本实验 pickle。 [analyze_caps.py](analyze_caps.py)

```sh
../.venv/bin/python -B handoff/codex/analyze_caps.py --self-test
../.venv/bin/python -B handoff/codex/analyze_caps.py \
  --model q35-4b --arms base,s8,s8rand,concise,reseed --caps 4096,8192,16384
```

缺文件或清单无该臂映射时，对应格留空。清单整体缺失时报告该状态。题号集合不等时，单臂统计仍可显示，配对格留空并给出原因。单侧 p 的方向是目标臂更准，`s8` 对 `s8rand` 的参考臂为 `s8rand`。每答对一题的 token = 所有回答的输出 token 总量 ÷ 答对题数，包含失败和截断回答的消耗，prompt 和重复预填的成本另计。无完成答案时，完成回答准确率记为 JSON `null`。无答对时，每答对一题的 token 也记为 `null`。各档共享历史前缀，汇总表的 CI 对应该档各自的 base 差值。脚本通过隔离的 `common.OUT` 导入原 `analyze.py`，复用其 bootstrap、转移和 token 函数。 [analyze_caps.py:21–51](analyze_caps.py#L21) · [analyze_caps.py:76–168](analyze_caps.py#L76)

合成自测已通过，覆盖 3×3 转移、错序题号、单侧和双侧检验、缺格和无分母情况。另用真实 1.7B 清单通过了 CLI、Markdown 和严格 JSON 检查，4096 档的 400 题 base 准确率和 edit 胜负与原清单一致。两份报告均写入临时目录进行验证，期间禁止导入 MLX。 [analyze_caps.py:241](analyze_caps.py#L241) · [1.7B 清单](../../out/mech_1.7b_summary.json)
