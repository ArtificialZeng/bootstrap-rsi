# 独立代码审查：`agent8b.py` / `common.py`

**审查对象**：`exp/agent8b.py`（556 行）、`exp/common.py`（127 行）
**对照基准**：`handoff/protocol_spec.md`（arXiv:2611.21885 v2 的协议规格，下文记作 **PS**，引用其 §/Tab./Alg. 编号）
**审查方式**：静态阅读 + 与已安装依赖源码核对（mlx 0.32.4、mlx_lm 0.32.2、math_verify 0.9.0）+ 读 `out/8bsmoke.log`、`out/8b_session8bsmoke_1.json` + 两组 CPU 级 `math_verify` 语义/信号实验。**未加载任何模型、未改任何代码、未安装任何东西。**

## 结论摘要

1. **正常路径下编辑不会泄漏进编辑者自己的推理**：编辑者的 `lm_generate` 在任何 `edited()` 之外（L444），所有工具内生成都在 `with self.subj.edited()` 里（L248-250、L260-261、L285-289、L299-300）。但"编辑者永不用编辑后权重"只是**调用顺序**保证的，`edited()` 的恢复逻辑本身很脆（BUG-04），而且下面的沙箱问题可以直接绕过它。
2. **受限 Python 会话不是沙箱**：任意代码块都能从工具函数拿到 `Env`（含 40 道**预留题的题干与 gold**）、直接改权重绕过 `submit()` 之后的锁、甚至拿到 `os`（BUG-01）。这是本次审查最严重的问题。
3. **按当前配置，`submit()` 的框架检查几乎必然超时**：检查要生成 2×40 个 ≤4096 token 的答案，而整个代码块只给 1500 s；按 smoke 实测吞吐推算，只有平均答案 ≲400 token 才可能跑完（BUG-02）。模型实际上无法提交。
4. **任何"编辑有效"的结论都混入了量化路径差**：编辑张量以 bf16 `nn.Linear` 替换 8-bit `QuantizedLinear`，base 臂走 `quantized_matmul`、edited 臂走 bf16 matmul；即使 W 不变两臂也不同（BUG-03）。
5. **评分口径与论文不同**：`math_verify` 会把多个不同 `\boxed{}` 解析成一个集合；实测 `\boxed{5} … \boxed{7}` 对 gold=7 判 **False**，而论文取"thinking 之后最后一个 boxed"（BUG-11）。这会让长答案更吃亏，方向不一定对称。

---

## 1. Bug

| ID | 严重度 | 位置 | 一句话 |
|---|---|---|---|
| BUG-01 | **严重** | L333-345、L180-181、L204-206 | 工具函数内省泄露 `Env`：可读预留题 gold、绕过权重锁、拿到 `os` |
| BUG-02 | **严重** | L293-304、L352-359、L492 | `submit()` 检查与代码超时/会话预算时间尺度不匹配，实际无法提交 |
| BUG-03 | **严重** | L54-62、L70-88、L98-108 | 8-bit 权重 → bf16 `nn.Linear`，两臂数值路径不同，增益混杂 |
| BUG-04 | **严重** | L81-83、L98-108 | `edited()` 按 `self.edits` 键恢复、`base` 在首次 `set` 时抓取：可泄漏编辑或永久污染 base |
| BUG-05 | 中 | L204-206、L243-290、L306-314 | 检查后仍可跑生成类工具；`withdraw()` 可覆盖已 `confirm()` 的决定 |
| BUG-06 | 中 | L478-483、L505-546、L149-151 | 缓存键只有 tag；`paired_stats` 用 `zip` 不校验长度，静默错配 |
| BUG-07 | 中 | L352-359 + common L54-63 | `math_verify` 内部 `signal.alarm` 会取消框架的代码超时；超时还可被用户代码吞掉 |
| BUG-08 | 中 | L228-231、L90-93 | `reset(name)` 不校验名字，拼错静默 no-op，后续测试仍带旧编辑 |
| BUG-09 | 中 | L267-290、L119-133 | `hidden_states` 不校验层号；`start >= len(seq)` 时返回 nan 方向 |
| BUG-10 | 中 | L501 | 预留集硬编码 `prac[60:...]`；`n_practice > 60` 时与练习题重叠 |
| BUG-11 | 低 | common L54-63、agent L141 | 多 box 判错（论文取最后一个 boxed）；EOS 恰在 cap 处误判截断 |
| BUG-12 | 低 | common L43、L50；agent L444 | 工具输出混入 throughput 打印；工具重置全局 RNG，影响编辑者采样 |
| BUG-13 | 低 | L451-452、L468 | `code_left<=0` 的提示永远读不到；没有"最后只做决策"的回合 |

### 先回答任务点名的几项

**（a）`edited()` 是否让编辑泄漏进编辑者推理？** 正常路径**不泄漏**。证据：编辑者回复的 `lm_generate`（L444）不在任何 `edited()` 上下文内；所有会跑权重的工具（`run` L248-250、`paired_test` L260-261、`hidden_states` L285-289、`submit` 的检查 L299-300、`main` 的评测 L528、L544）都在 `with self.subj.edited()` 内，`finally` 会恢复。**但是**：(i) 上下文是用 `__enter__/__exit__` 实现的，代码块里可以手动 `enter` 不 `exit`（BUG-01）；(ii) 恢复依赖 `self.edits` 的键和 `self.base` 快照，这两者在异常路径下都可能出错（BUG-04）。也就是说，这条性质目前是"靠调用顺序碰巧成立"，不是被强制的。

**（b）反量化是否正确？** `Subject.base_weight`（L54-62）调用 `mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)`。与 mlx 0.32.4 的签名 `dequantize(w, /, scales, biases=None, group_size=None, bits=None, mode='affine', …)` 一致；8-bit checkpoint 确实带 `scales`/`biases`（已核对 `model.safetensors.index.json`：`…o_proj.{biases,scales,weight}`）。形状为 `(out, in)`，`nn.Linear(in, out)` 的替换（L84）方向正确。**唯一问题**是量化值被转成 bf16 后走不同算子（BUG-03），以及 `base_weight` 返回的 W 与 base 臂实际用的 8-bit 矩阵有极小的重构差。

**（c）`weaken` 的方向是否正确？** 对 `o_proj (4096,4096)` 与 `down_proj (4096,12288)`：行维 = 残差维，`W ← W − α·u(uᵀW)`（L225）等价于对输出做 `(I−αuuᵀ)`，与论文 §2.2 / `common.edit_layer` 的定义一致，`u` 长度校验 `u.shape[0] == W.shape[0]`（L222）也对。**注意**：`gate_proj/up_proj`（行 12288）不是残差空间，系统提示却只说"u: direction in the space W writes to"（L389），模型完全可能在一个非残差空间里"削弱方向"并误以为在做论文的编辑；建议在提示里明确"只有 o_proj/down_proj 的行是残差流"。

**（d）`paired_stats` 是否正确？** δ̂ 与 SE 的公式（L151-153）与论文 Eq. 3（PS §2.2）逐项一致；单侧符号检验（L154-155）在 `d≠0` 的对上做，方向是"edited − base"，正确。问题在于：每对只有 1 个答案（多个答案时 `d_i` 应取均值，这里的输入天然是 1 答案）、`zip` 不比长度（BUG-06）、`n=1` 时 SE 为 nan、截断率只报均值不报 SE（论文的报告里有截断的 δ̂/SE）。**优点**：它额外算了 `base_acc_finished`/`edit_acc_finished`（L164），正好补上论文没做的"截断 vs 答完变准"分解（PS §7）——建议保留并写进结果表。

**（e）预算/状态机有没有漏洞？** 计费本身符合论文：`run`/`paired_test`/`hidden_states` 各 1 单位（L245、L256、L283）、检查不扣费（L298-301）、起始诊断不扣（PS §3.1）；检查的回报只有聚合统计（`submit()` 直接调 `paired_stats`，L301，不含 `per_problem`），与论文一致，这点是对的。漏洞在**锁**：`_locked()`（L204-206）只挡 `Env.set_weight/weaken/reset`，而 `Subject.set` 本身没有锁，且可从代码块直接拿到（BUG-01）；检查后 `run/paired_test/hidden_states` 仍可跑（BUG-05）；`withdraw()` 可覆盖 `confirm()`（L309-314）。

**（f）沙箱有没有绕过？** 有，且是平凡的（BUG-01）。`ns` 里放的是**绑定方法**（L338），所以 `run.__self__` 就是 `Env`，`run.__func__.__globals__` 就是 `agent8b` 模块全局（含 `os`、`sys`、`signal`）。`SAFE_BUILTINS` 还包含 `object`/`type`，`object.__subclasses__()` 这条路也不需要 `getattr`。此外 `ns["np"]` 是 numpy 模块，`np.fromfile`/`np.save`/`np.ctypeslib` 直接读写文件/加载动态库。系统提示里"No file or network access"（L385）与实际不符。

---

### BUG-01　【严重】受限 Python 会话可拿到 `Env`（含预留题 gold），并绕过权重锁

**行号**：L333-345（`make_namespace`；关键在 L338 `ns[f] = getattr(env, f)`、L343 `lambda pid: env.practice[pid]["problem"]`）；相关 L180-181（`self.reserved = reserved`）、L204-206（锁）、L298-301（检查用的就是 `self.reserved`）。

**触发**（任选其一，都在普通代码块里）：
- `run.__self__.reserved` → 40 道预留题的完整 dict（含 `problem` 与 **`answer`**）；`run.__self__.reserved[0]["problem"]` 就是检查题面。
- `run.__self__.subj.set("layers.8.mlp.down_proj", W)` → 绕过 `Env._locked()`：`submit()` 之后仍能改权重，`confirm()` 前再换一个编辑，检查结果与最终保存的编辑不是同一个。
- `run.__self__.model` → 直接跑模型、不花 B。
- `problem.__closure__[0].cell_contents` → 同一个 `Env`。
- `run.__func__.__globals__["os"].system(...)` / `sys.modules` → 完全逃逸；`np.fromfile/open` 可读 `~/.cache/huggingface/.../MATH-500/test.jsonl`（全部评测题与 gold）和 `out/8b_eval_base.pkl`。

**后果**：
1. 检查集（40 道预留题）的题面和 gold 对模型可见——而检查是提交前唯一的筛选，协议 2 的"预留题会话不可读不可用"（PS §4）失效；模型可以专门针对检查集调编辑，甚至用 gold 直接构造编辑。
2. `submit()` 后的"编辑已锁"（L206 的承诺）可被绕过，`confirm()` 保存的编辑与检查过的编辑不同。
3. 系统提示的"no file or network access""no gradients"等约束只是**提示**，不是强制；8B 模型若偶然用上 `np.save`/内省，实验会跑出不可解释的结果。
4. 任何"受限会话"的行为学结论（例如"模型没有作弊"）都不成立。

**建议改法**：
- **不要在执行命名空间里放任何绑定方法或闭包**。代码执行与模型/权重必须分进程：子进程只有 numpy + 一个 JSON 行协议（`{"op":"run","ids":[...]}` → 父进程生成 → 回传 `{id,correct,truncated,ntok,ending}`），父进程持有 `Env`、`Subject`、`model`、`reserved`。子进程用 `resource`/`SIGKILL` 限制。
- 若坚持单进程，至少：(i) 用模块级 `def _tool_run(ids): …` 只传数据（但 `__globals__` 仍可逃逸，**单进程 Python 无法真正沙箱**，只能算提高门槛）；(ii) 把 `reserved` 从 `Env` 里移到 `main` 的局部变量、检查时再传参；(iii) `Subject.set` 加与 `Env._locked` 等价的锁，或把锁放进 `Subject`；(iv) 在提示里删掉"no file access"这类不实的承诺。

---

### BUG-02　【严重】`submit()` 的检查与实际时间尺度不匹配（模型无法提交）

**行号**：L298-304（检查：2×40 个答案、`cfg.cap`=4096）、L352-359（`exec_code` 给整个代码块 `signal.alarm(1500)`）、L492（`code_timeout=1500`、`hours=3.0`、`B=10`）。

**触发**：模型调用 `submit()`（这是提交的唯一入口）。

**后果**：
- smoke 实测（`out/8bsmoke.log`）：8B 8-bit 在这台机器上约 **21 tok/s**（batch 6、384 token）。检查要生成 80 个答案；按 21 tok/s，**80 个答案平均只要超过约 390 token 就会超过 1500 s**——也就是说几乎一定超时。即使批量放大到 60–100 tok/s，平均答案超过约 1100–1900 token 也会超时；MATH-500 在 4096 cap 下的平均长度远大于此。
- 超时会在**生成中途**抛 `_Timeout`（L364-365），`self.checked` 永不置位（L302），模型只看到一条 TimeoutError；它可以再 `submit()` 一次，再烧掉 ~1 小时。会话 3 h 的预算基本被这一步吃光。
- 同样的算术也压到其他工具：`paired_test` 是 2×16 个答案（L259-261），临界；`run` 是 16 个答案（L250）。10 次 pass × 每次 20–60 分钟，远超 3 h，多数会话会以 `withdraw (no decision)`（L471）结束——这不是模型的决策，而是超时的产物。
- `_Timeout` 在 MLX 生成中间抛出，模型/流状态未定义，后续生成是否正常未经验证。

**建议改法**：
- 把检查移出代码执行超时：作为框架步骤（模型 `submit()` 只做标记，框架在会话循环里跑检查、再让模型 `confirm/withdraw`），并给检查单独的、按"答案数 × cap ÷ 实测 tok/s"算出的超时。
- 按实测 tok/s 反推每次测试的题数（例如 `run`/`paired_test` 6–8 题，检查 10–16 题），或在配置里显式给 `--tok-s` 并自动裁。
- 超时后记录"部分完成"的答案而不是丢弃；`submit()` 失败要能被再次调用且不重复烧预算。
- 提高 `hours` 或降低 `B`，让"跑完一次会话"在物理上可能。

---

### BUG-03　【严重】编辑以 bf16 `nn.Linear` 替换 8-bit `QuantizedLinear`，base/edited 两臂数值路径不同

**行号**：L54-62（反量化）、L84-87（`nn.Linear(W.shape[1], W.shape[0], bias=False)` + bf16 权重）、L98-108（安装/恢复）；对照 mlx 0.32.4 `QuantizedLinear.__call__`（`mlx/nn/layers/quantized.py:272`，用 `mx.quantized_matmul`）。

**触发**：任何 `set_weight` 之后跑 `run`/`paired_test`/`hidden_states`/评测。

**后果**：
- base 臂对同一矩阵用 8-bit `quantized_matmul`（int8 权重 + scales/biases 重构），edited 臂用 bf16 matmul（反量化→bf16 舍入）。**即使编辑是 W′=W，两臂的输出也不会逐位相同**。因此 `paired_stats` 测到的是"编辑 + 量化路径差"的合成效应。
- 生成对数值差异极敏感（一个 token 翻转会改变整条轨迹），所以这个"路径差"不是可以忽略的噪声；它可能造出几个百分点的假增益/假损失。对"编辑是否让模型变好"这一核心结论是致命的。
- 反量化算出的 `base_weight` 与 8-bit 实际权重有极小差，`weaken`/`set` 的 W 因此不是模型真正用的矩阵；`summary()` 的"相对变化"也是相对重构矩阵。
- **这条路径在 smoke 里从未被端到端跑过**：smoke 只在 turn 3 成功 `set` 了 `layers.0.self_attn.o_proj`（`edits` 非空），随后 turn 4 是"thinking 超长、无代码块"，会话到 4 轮就结束（见 `out/8b_session8bsmoke_1.json` 与 `8bsmoke.log`）——没有任何 `run`/`paired_test` 真正*用*编辑后的权重生成过。所以 bf16 `nn.Linear` 的 forward、dtype 提升、内存都还没被验证，这类错误只会在正式跑时暴露。

**建议改法**（任选其一，优先级从高到低）：
1. 把编辑后的 W 用 `mx.quantize(W, group_size=..., bits=8)` 重新量化，替换成同样配置的 `QuantizedLinear`——两臂都走 `quantized_matmul`，路径一致。
2. 两臂都用 bf16：加载后把整个模型反量化成 bf16（8B 在 128 GB 机器上可行），再做所有实验——代价是显存/速度。
3. 至少加一个 **no-op 对照**：对同一批张量安装 W′=base_weight(W)（bf16 路径、数值不变），测一次 held-out；这个 delta 就是纯路径差，应当从所有结果里扣掉/报告。

---

### BUG-04　【严重】`edited()` 的恢复逻辑脆弱：可把编辑泄漏进编辑者，或永久污染 base

**行号**：L81-83（第一次 `set` 时 `self.base[path] = getattr(par, a)`）、L98-108（`edited()` 按 `self.edits` 的键安装/恢复）。

**触发**：
- 在 `edited()` 活动期间调用 `set()`/`reset()`（正常 API 不会嵌套，但 BUG-01 的 `subj.edited().__enter__()` 可以做到；未来重构也可能）；
- `__enter__` 期间安装循环中途抛异常（L100-102 在 `try` 之前，已安装的模块不会被恢复）；
- 两层 `edited()` 嵌套。

**后果**：
- `self.base[path]` 若在编辑已安装时被写入，保存的就是**编辑后的模块**；此后每次 `__exit__` 都把"编辑"当原权重恢复——模型被永久改坏，且所有后续测量都错，且没有任何报错。
- 退出时只遍历当时的 `self.edits` 键：若在上下文内 `reset()` 了某个路径，该模块永远不会被恢复 → **编辑留在模型上**，编辑者下一次推理就用编辑后的权重，直接违反论文"编辑者自身权重不变"（PS §2.2）的核心设定。
- 手动 `__enter__` 后不退出：同样让编辑者带着编辑推理，并且绕过"检查后锁定"。

**建议改法**：
- `__enter__` 时先快照 `[(parent, attr, original_module)]`（而不是依赖 `edits` 的键），`__exit__` 按快照恢复；禁止嵌套（进入时置 flag，重入直接抛错）。
- `Subject.__init__` 时一次性遍历所有可编辑路径保存原始模块（不可变），`base` 不再在 `set` 时抓取。
- 把"编辑者永不用编辑权重"做成断言：在 `agent_session` 生成前 `assert not any(isinstance(getattr(...), nn.Linear) for ...)`，或让 `edited()` 只在受限的子进程/线程里生效。

---


### BUG-05　【中】检查后生成类工具不受限；`withdraw()` 可覆盖已确认的决定

**行号**：L204-206（锁只对权重写操作生效）、L243-290（`run`/`paired_test`/`hidden_states` 不检查 `self.checked`）、L306-314（`confirm`/`withdraw` 都直接写 `self.decision`）、L468（循环出口）。

**触发**：`submit()` 之后继续 `run`/`paired_test`；或一个代码块里 `confirm(); withdraw()`。

**后果**：
- 检查后还能花预算/时间做新测试（权重虽然锁了，但会话时间和 B 是共享的），可能把会话拖到超时；论文的语义是"检查之后模型只回答 SUBMIT 或 WITHDRAW"。
- `submit(); confirm(); withdraw()` → 最终 `decision="withdraw"`，一个已通过检查的编辑被静默丢掉；反之 `withdraw(); confirm()` 会因 `checked is None` 抛错（L307-308），状态不一致。
- 检查后若超时/轮数到而未 `confirm`，`decision=None` → `withdraw (no decision)`，等于丢编辑。

**建议改法**：`confirm`/`withdraw` 之后立即结束会话（`raise` 一个 `SessionEnd`，或置 `final` 标志并在循环顶部 break）；检查后禁用 `run/paired_test/hidden_states`（`_locked` 扩展到只读之外的生成类工具）；`decision` 一旦置位不再允许改写。

---

### BUG-06　【中】缓存键只看 tag；`paired_stats` 用 `zip` 不校验长度 → 静默错配

**行号**：L478-483（`cached(path, fn)` 只按路径存在与否）、L505-510（practice_base/eval_base）、L530（eval_script）、L546（`eval_session{s}`）；L149-151（`zip(base, edit)`）。

**触发**：改 `cfg.cap`/`n_practice`/`n_eval` 后用同一个 tag 重跑；或重跑同一 session（编辑不同）但 `eval_session{s}.pkl` 已存在。

**后果**：
- `base_prac`/`base_eval` 用旧 cap 的答案，和新的 edited 答案配对（截断判定、长度分布全不同）→ 结果错但不报错。
- 重跑 session s 会复用**上一次**编辑的评测结果，`results` 里写的是新编辑、数字是旧编辑的。
- `paired_stats` 若两列表长度不同（例如 `n_eval` 改小但 `base_eval` 缓存是旧的 160），`zip` 静默截到较短者，且不检查 `pid` 是否对齐——配对检验会拿不同题目的答案比。

**建议改法**：cache 键加入 `(tag, cap, n_practice, n_reserved, n_eval, 编辑摘要)`（编辑摘要可用 `subj.summary()` 的哈希）；`paired_stats` 开头断言 `len(base)==len(edit)` 且 `[r["pid"] for r in base] == [r["pid"] for r in edit]`。

---

### BUG-07　【中】`math_verify` 的 `signal.alarm` 会取消框架的代码超时；超时还能被用户代码吞掉

**行号**：L352-359（`exec_code` 的 SIGALRM）、L348（`_Timeout(Exception)`）、common L54-63（`grade` → `math_verify.parse/verify`，两者内部都用 `signal.alarm`）。

**触发**：任何在代码块里先 `run()`/`paired_test()`（内部逐答案 `grade`）再写长循环/死循环的代码；或 `try/except Exception` 包住长循环。

**后果**（本机 CPU 实测确认）：`math_verify.utils.timeout` 在进入时 `signal.alarm(t)`、在 `finally` 里 `signal.alarm(0)`。由于一个进程只有一个 alarm，**第一次 `parse()`/`verify()` 就会把框架的 1500 s 计时清掉**（实测：先设 60 s，调用一次 `parse` 后再读 pending alarm = 0）。于是：
- 在 `paired_test` 里，base 臂的生成受超时保护，edited 臂的生成（在 base 的评分之后）**完全不受保护**；更一般地，**同一代码块里只要先调用过任何带评分的工具（`run`/`paired_test`），这个块剩下的部分就再也没有框架超时**；
- 模型可以写一个 `try: while True: pass except Exception: pass`，`_Timeout` 是 `Exception` 子类（L348），被吞掉后 alarm 已取消，**整个实验挂死**（没有进程级 watchdog）。
- `exec_code` 的 `except Exception` 虽在前面有 `except _Timeout`，但用户代码内部先吞掉就轮不到它。

**建议改法**：不要用进程级 SIGALRM 做超时。用子进程 + `SIGKILL`（同时解决 BUG-01），或用"时间戳轮询"包装每个工具调用；若保留 SIGALRM，则在每次工具调用返回后重新 `signal.alarm(剩余时间)` 并在 `grade` 前后保存/恢复剩余时间。`_Timeout` 改为继承 `BaseException`，避免被 `except Exception` 吞掉。

---

### BUG-08　【中】`reset(name)` 不校验名字，拼错静默 no-op

**行号**：L228-231（`Env.reset` → `subj.reset`）、L90-93（`Subject.reset` 只 `pop`）。

**触发**：`reset("layers.8.mlp.down_porj")`（拼错）或 `reset("layers.8")`。

**后果**：模型以为自己撤销了某个编辑（返回 "reset layers.8.mlp.down_porj"），实际什么都没发生；下一次 `paired_test` 仍带着旧编辑，模型会把"旧编辑 + 新编辑"的效果归因到新编辑上，得出错误诊断。这个错误对 8B 模型很现实（它在 smoke 里就写错过参数名）。

**建议改法**：`reset` 先 `self.subj._check(name)`；名字不在 `self.edits` 时返回明确错误（或返回"No such edit；current edits: [...]"）。

---

### BUG-09　【中】`hidden_states` 不校验层号与空切片

**行号**：L267-290（`layers` 只排序、不查范围）、L119-133（`out` 只填 `i in layers`；`h[0, start:]`）。

**触发**：`hidden_states(layers=[40])`（模型有 36 层，0–35）；或 `len(prompt_ids) >= 6000`。

**后果**：
- `layers=[40]`：`Lmax=40`，循环到 35 结束，`out` 里没有 40 → `h[l]` 抛 `KeyError: 40`（被 `exec_code` 捕获成一条不指向原因的报错），模型白花 1 单位。
- `start >= len(seq)`：`h[0, start:]` 是空切片，`.mean(axis=0)` 返回 **nan 向量**；模型可能把它当方向用，`set_weight` 因非有限值拒绝（L76-77），但错误信息与真实原因无关；`hidden_states` 的 1 单位也白花。

**建议改法**：`0 <= l < len(model.model.layers)` 与 `start < len(seq)` 校验并给出明确错误；`layers` 上限用模型层数。

---

### BUG-10　【中】预留集硬编码在 `prac[60:...]`

**行号**：L501：`practice, reserved, ev = prac[:cfg.n_practice], prac[60:60 + cfg.n_reserved], ev[:cfg.n_eval]`。

**触发**：任何 `n_practice > 60` 的配置（例如把练习扩到 100 题，或改 `n_practice` 但不改这个 60）。

**后果**：`practice` 与 `reserved` 重叠 → 模型在练习题里直接看到检查题（`practice_ids`、`base_results`、`problem()`），检查失去意义；更糟的是它可以用练习题调参后在"同题"上检查，结果虚高。当前的默认 60/40 恰好不相交，所以这是一个**潜伏的配置陷阱**。

**建议改法**：`prac[cfg.n_practice:cfg.n_practice + cfg.n_reserved]`，并在启动时断言 `practice ∩ reserved = ∅`、`(practice ∪ reserved) ∩ eval = ∅`。

---

### BUG-11　【低】评分口径偏离论文：多 box 判错、EOS-at-cap 误判截断

**行号**：common L54-63（`grade`：`ntok >= max_tokens`、`text.split("</think>")[-1]`、`verify(parse("$"+gold+"$"), parse(ans))`）；agent L141。

**触发**：答案里出现两个不同的 `\boxed{}`；或模型恰好在 cap 处输出 EOS。

**后果**（本机 CPU 实测，math_verify 0.9.0）：
- `parse(r"\boxed{5} then \boxed{7}")` → `[{5, 7}, '5,7']`，`verify(7, …)` = **False**；`\boxed{7} then \boxed{5}` 同样 False。论文的规则是"thinking 之后**最后一个** `\boxed{}`"（PS §5.2），这里会把"先写错、最后改对"的答案判错。
- 重复写同一个 box（`\boxed{7}…\boxed{7}`）会得到 `[7,'7']`，判对；所以偏差只发生在**出现多个不同中间 box** 时。编辑的目标正是减少重复/多步"验算"，base 臂的长答案更容易出现多个 box → 该偏差**不完全对称**，可能偏向 edited 臂。
- `ntok >= max_tokens` 把"恰在 cap 处正常结束"也算截断（很少见，但方向是把对的算错）。
- 没有 box 时 `parse` 会从正文里抽数字（实测 `"first 5, second 6, final 7"` → `7`，判对），所以"只看 boxed"并不严格；影响较小，但和论文口径不完全一致。

**建议改法**：在 thinking 之后自己扫最后一个平衡括号的 `\boxed{...}`，只对那一个调用 `parse/verify`；截断用生成结束原因（是否 EOS）而不是 token 数；把 `parse` 的 `extraction_mode` 固定为 `first_match` 只在 box 子串上使用。

---

### BUG-12　【低】工具输出混入 throughput 打印；工具重置全局 RNG 影响编辑者采样

**行号**：common L43（`mx.random.seed(seed)`）、L50（`print("  generated … tok/s")`）；agent L444（编辑者 `lm_generate` 未显式 seed）。

**触发**：每次工具内的生成。

**后果**：
- `exec_code` 把 stdout 重定向到 buf（L361），所以 `common.generate` 的 "generated 16 x <= 4096: … tok/s" 会原样进入模型看到的工具输出，挤占 4000 字符窗口、制造噪声。
- 工具生成 `mx.random.seed(500x)` 会**重置全局 RNG**，而编辑者下一次 `lm_generate` 没有显式 seed，于是编辑者的采样取决于"上一次工具调用用的种子"。结果：同一 session 不可复现；打开/关闭某个工具会改变 agent 的后续行为，跨条件比较被污染。

**建议改法**：`generate` 加 `verbose=False` 默认；编辑器每轮 `mx.random.seed(hash((session, turn)))`；工具生成前后保存/恢复 `mx.random.state`（或在独立进程里生成）。

---

### BUG-13　【低】`code_left<=0` 分支的提示永远读不到；没有"最后只做决策"的回合

**行号**：L451-452（文案本身有病句："Call submit()/confirm() or withdraw() is no longer possible; the session ends."）、L454（不执行代码）、L468（`if env.decision or code_left <= 0: break`）。

**触发**：`C <= turns`（smoke 是 C=6/turns=4，不触发；默认 C=60/turns=50，也不触发），或未来把 C 调小。

**后果**：这一轮不执行代码，却把"session ends"写进 feedback，然后立即 break——模型永远看不到这条消息；如果它本来还有编辑想 `confirm()`，就被剥夺了机会。

**建议改法**：配额用尽时进入一个"final turn"（只允许 `submit/confirm/withdraw`，不运行代码），跑完再结束；或删掉这条不可达分支。

---

### 其他观察（未列为 bug）

- `Subject.summary()` 每轮调用（L463），每次都 `base_weight()` 反量化（`down_proj` 约 50M 参数），一个 session 几十轮会反复分配数百 MB 并拖慢每轮；建议缓存反量化的 base。
- `script` 阶段若 `H[True]` 或 `H[False]` 为空会**静默跳过**（L523-530），`results` 里就没有 `script`，不报错。
- `json.dump(..., default=str)` 会把 `checked` 里的非 JSON 类型转成字符串（L466-467、L550）；下游分析要重新解析。`paired_stats` 在无已答完的答案时 `acc_finished=nan`，JSON 里写 `NaN`（非严格 JSON）。
- 内存：`common.generate` 默认 `completion_batch_size=64`（common L46），4096 token 的 8B KV 约 0.6 GB/序列，64 并发约 38 GB；本机 128 GB 可以，但换机器会 OOM；建议 `bs=8–16` 或按 `n_prompt` 收紧。

---

## 2. 与论文协议的偏差

按 PS 的条目对照（已按要求略去"练习 60 题 / 预留 40 题×1 答案 / B=10 / MATH-500 / 回复 6000 token"这些已知的有意缩小）。

| # | 项目 | 代码 | 论文（PS 出处） | 影响判断 |
|---|---|---|---|---|
| 1 | **模型与规模** | Qwen3-8B（Qwen3 家族，8-bit MLX） | Qwen3.5 家族 27B（PS §3.1；论文自己报告同族 9B "break themselves"） | **实质**。架构不同（Qwen3 标准 GQA vs Qwen3.5 的 Gated DeltaNet 混合），可编辑张量集合不同；论文的核心主张是"能力随规模出现"，8B 的负结果不能反驳、正结果也不能外推。任何结论只能表述为"Qwen3-8B 上的一次小规模复现"。 |
| 2 | **量化与编辑表示** | 8-bit 权重 + bf16 `nn.Linear` 替换（BUG-03） | bf16 权重、`W ← W − αuuᵀW`（PS §2.2/§2.3） | **实质**。base/edited 数值路径不同，delta 混入量化路径差；在 8B + 4096 cap 的混沌生成里，这个混杂可能大于编辑本身。 |
| 3 | **检查回报的内容** | 聚合统计（`paired_stats`：δ̂、SE、wins/losses、截断率、acc\|finished、p），**不含**逐题明细 | 聚合"点数 + SE"（PS §4、Tab. 13） | 轻微。比论文只多不少、且不泄漏预留题；额外的截断率/acc\|finished 是有用的诊断量（p 的检验方式见 #6）。 |
| 4 | **一次 pass 并测多个编辑** | 只能测"当前累积编辑" | 一次前向可并排测最多 4 个已保存编辑（PS §4） | **实质**（结构）。协议 2 的核心效率机制缺失：模型无法在同一预算下比较候选，只能 reset 再测，10 次 pass 根本不够。 |
| 5 | **种子** | 每次调用 `self.seed += 1`（L186/L247/L258） | 按题目 id 哈希固定、base/edited/各 pass 相同（PS §2.3、Tab. 5） | 中等。同一题在不同 pass 用不同种子，模型"重复测同一批题"的证据无法干净比较；评测仍配对（同一次调用内同种子），所以评测本身没错。 |
| 6 | **评测设计与统计** | 160 题×1 答案；单侧符号检验（L155） | 490 题×8 答案；预注册的符号翻转置换检验（100,000 次）、固定阈值、session 8 两次查看（PS §5.2、§6） | **实质**。数字不可与 Tab. 8/13 比较；160×1 的功效低；未预注册、无多重比较控制。符号检验本身有效，但丢弃了"同题 8 答案"的相关结构。 |
| 7 | **编辑者的历史** | 只存 `</think>` 之后、≤3000 字符（L457-460）；thinking 丢弃；无 `notes.md`、无 90k 重启协议 | 完整回复进 h；`notes.md` 在重启后重新展示（PS §1.2、§3.1） | 中等。模型看不到自己此前的推理与数字（比如"上一轮算出的方向、alpha"），行为与论文里的 27B 会话不同；长会话后期尤其明显。 |
| 8 | **模型能看到的答案** | `run` 只给 300 字符结尾；无编辑臂完整答案；`answer_text` 只给 base；**不给参考答案** | practice 题带参考答案；所有答案落盘可读（PS §1.2 工具(2)、Alg. 3 返回） | 中等。模型无法核对 gold、无法细读自己/编辑后的失败文本——论文的诊断循环（"看自己的错答"）被削弱；也会降低它找到有效编辑的概率。 |
| 9 | **hidden states 接口** | 只有"平均残差"，≤64 项、6000 token | 可选 hidden states / MLP 激活 / log-prob、多种聚合、≤256 文本×8192（PS §2.2/§2.4） | 轻微–中等。方向类编辑仍可做；但论文里"读 MLP 激活定位损坏矩阵"一类诊断无法复现。 |
| 10 | **prompt** | 自写 system+user 两条；明确给出形状表与"o_proj/down_proj 的行是残差流" | 单条消息、转述版（PS §1）；论文本身也没有逐字 prompt | 中等。这里的提示比论文的转述更"直给"（形状、残差流、weaken 公式），降低了任务难度，也可能把模型引向论文的成功路径；跨模型比较时要记住这一点。 |
| 11 | **submit/confirm 流程** | `submit()` 触发检查并返回结果，模型再 `confirm/withdraw`；检查后仍可跑工具（BUG-05） | 框架先测，模型只答 SUBMIT/WITHDRAW（PS §4） | 轻微–中等。语义接近，但"检查后还能继续测"和"decision 可被覆盖"改变了决策动力学。 |
| 12 | **在用模型曲线** | 每个 session 从 base 起、单独评测提交的编辑 | 同样单独评测；另外 Fig. 1a 还画了"提交即替换在用模型"的累积曲线 | 轻微。没复现累积曲线，不影响单编辑结论。 |
| 13 | **报告完整性** | 默认 `--sessions 2`；无预注册 | 报告全部 8 个 session（含 5 次撤回） | 中等。若只跑 2 个并挑好的报告，就是选择性报告；应固定"跑 N 个全报"，包括撤回。 |
| 14 | **脚本基线** | 论文没有的额外对照：默认用 1.7B sweep 的层/α（6 / 1.0），有 `out/summary.json` 时读它；层号按 `L17/27×(n_layers−1)` 比例迁移到 8B；方向用 8B 自己的 practice 估 | ——（论文无此基线） | 中等。作为"非智能体基线"是加分项，但 L/α 从 1.7B 迁移、α=1.0 是全额移除，可能比论文的 5% 编辑强得多或弱得多；它只能说明"这个特定脚本在 8B 上如何"，不能替代论文缺失的基线比较。 |

**会实质影响结论的偏差**：#1（模型/规模）、#2（量化混杂）、#4（无并排测试）、#6（评测与统计不可比）、#13（报告完整性）；#5/#7/#8/#10 会让 agent 的行为与论文不同，影响"复现"二字的解释力。

---

## 3. 可选改进（≤5，按价值排序）

1. **同精度对照（最高优先）**：把编辑重新 `mx.quantize` 回 8-bit（或两臂都用 bf16），再加一个"no-op 编辑"对照，报告"纯量化路径差"。没有这一步，任何 +x 分都不能归因于编辑。
2. **把检查移出代码超时，并按实测吞吐决定 pass 规模**：用一次 smoke 测 tok/s，自动把 `run/paired_test` 的题数、检查题数、`hours` 调到能跑完；检查作为框架步骤、给独立超时。否则模型永远提交不了（BUG-02）。
3. **改用论文口径评分并输出分解**：只取 thinking 后最后一个平衡括号的 `\boxed{}`；截断用 finish reason；同时报告 `acc | finished`、截断率、`win/loss`（代码已有 `acc_finished`）。这既修 BUG-11，也正好补上论文没做的"截断减少 vs 答完变准"分解（PS §7）。
4. **工具搬到子进程 + JSON 协议**：父进程持模型/权重/预留题，子进程只有 numpy；超时用 SIGKILL。一次解决 BUG-01（泄漏/锁绕过）、BUG-04（手动 enter）、BUG-07（alarm 互相取消/挂死）。
5. **可复现性护栏**：cache 键加 config+编辑摘要；种子按题 id 哈希；每轮编辑器显式 seed 并保存/恢复 `mx.random.state`；`paired_stats` 断言长度与 pid 对齐；每个 session 的 `edits` 摘要、check、eval 一起落盘（含撤回的 session）。

---

## 附：本次审查做过的验证

- 读 `agent8b.py`、`common.py` 全文（逐行），读 `out/8bsmoke.log`、`out/8b_session8bsmoke_1.json`、`out/8bsmoke_results.json`。
- 核对已安装源码：`mlx/nn/layers/quantized.py`（`QuantizedLinear.__call__` 用 `quantized_matmul`；`mx.dequantize` 的 `scales/biases` 参数）、`mlx/core/__init__.pyi`（`dequantize` 签名）、`mlx_lm/generate.py`（`generate`/`batch_generate`/`BatchResponse`/`sampler`）、`mlx_lm/models/qwen3.py`（36 层、`TransformerBlock(x, mask, cache)`、`hidden_multi` 与模型 forward 等价）、`mlx_lm/models/base.py`（`create_attention_mask` 返回 `"causal"` 字符串，MLX 接受）。
- 核对 8-bit checkpoint：`config.json`（36 层、hidden 4096、intermediate 12288、group_size 64、bits 8）与 `model.safetensors.index.json`（每层 `{weight,scales,biases}`）。
- 两次 CPU 级 `math_verify` 实验（未加载模型）：多 box 语义（`\boxed{5} … \boxed{7}` 对 7 判 False）、以及 `parse` 会取消外层的 `signal.alarm`（实测 pending alarm 变 0）。
- 未运行任何模型、未下载、未安装、未改代码。
