# 复审：修复后的 `agent8b.py` + 新 `mech.py` / `common.py` / `queue.sh`

**审查对象**：`agent8b.py`（666 行）、`mech.py`（158 行）、`common.py`（153 行）、`queue.sh`（11 行）；对照上一轮 `review_agent8b.md` 与 `protocol_spec.md`（arXiv:2611.21885 v2，下文 **PS**）。
**审查方式**：逐行静态阅读；核对已安装依赖（mlx 0.32.4 / mlx_lm 0.32.2 / math_verify 0.9.0 / numpy 2.5.5）；用纯 CPU 小实验验证了 6 处行为（见附录；numpy 2.5.5）；只读了 `out/` 里的日志与 summary，**未修改任何文件、未运行模型、未占 GPU**（审查时 queue 正在跑 `mech.py --model 1.7b`，PID 62659，未干扰）。

## 结论摘要

1. **13 条旧 BUG：10 条修好（其中 07 修好了原问题，但引出了 N1），3 条部分修好（01/06/11）。** 修复质量整体很高：`Subject` 的不可变快照 + `assert_clean()`、`_tool` 的暂停/恢复、检查移出代码超时、`last_boxed`、`paired_stats` 的对齐断言、`main` 的切片断言都是正确方向。
2. **新引入的 N1 最危险**：`_tool` 用 `signal.alarm(rem)` 恢复计时，而 `alarm(0)` 返回的是**向上取整的整数秒**；只要模型在纯 Python 里每 <1 s 调用一次工具（`budget()` 就是工具），计时永远不减少。CPU 实测：2 s 超时下，每 10 ms 调一次 `budget()`，跑了 6 s 仍未超时。再加上 **bare `except:` 仍能吞掉 `_Timeout`**，`while True: budget()` 能把整条 queue 挂死（`queue.sh` 没有 watchdog）。
3. **`agent_session` 的 final 回合有陷阱**：如果模型直到 final 标记后的**下一轮**才第一次 `submit()`，检查会跑完，但循环用 `log[-2].final_turn` 立刻 break，模型**没有机会 confirm** → 检查通过的编辑仍被记为 withdraw。
4. **mech 的统计框架方向对，但还不足以支撑"增益只来自截断减少"**：base/edit/random/concise/reseed/续写六臂齐全，但每臂 1 样本、random 方向未匹配扰动大小、没有 edit@2×、`acc|finished` 是选择后的量、`base_acc_with_2x_budget` 无 CI；另外 mech 的缓存键不含 `n_eval`/`dirs`/`practice`，改 CLI 会静默复用旧结果。
5. **媒体报告层面最缺**：多编辑/多采样重复、机制分解的 CI、对照的直接配对检验、第二个 held-out 集、以及可核查的复现包与协议偏差声明（8B、bf16、400×1、40/40 检查 vs 论文 27B、490×8、300/256）。

---

## A. 旧 BUG 逐条

| ID | 判定 | 证据（行号） | 残留 / 说明 |
|---|---|---|---|
| BUG-01 沙箱 | **部分修复** | `Env` 不再持有 reserved：L234、L480、L491；`__` 过滤：L392-393；`SAFE_BUILTINS` 去掉 `object/type/hasattr`：L386-391；`make_namespace`：L402-414 | **预留题确实移走了 ✓，内省路径被 `__` 挡 ✓**；但 **numpy 文件/网络旁路仍可用**（见下） |
| BUG-02 检查超时 | **已修** | 检查移到 `exec_code` 之外：L480-488、L536-540；检查时间从会话时钟扣除：L497、L538、L542 | 无超时、无重试、无异常处理（见 N3）；queue 总时长被拉长 |
| BUG-03 量化路径 | **已修** | 加载时一次性反量化全部投影：L35-50、L576；mech 同样：mech L50 | 8B 的 base 现在是 **bf16 全模型**（16 GB），与上一轮 8-bit 数字不可比；报告中必须写清 |
| BUG-04 `edited()` | **已修** | 构造时快照 `(parent, attr, module)`：L59-64；`_guard`：L84-88；`edited()` 用 `installed` 恢复 + 禁重入：L121-137；`assert_clean`：L139-142；生成前调用：L520 | `assert_clean` 失败直接 raise，本轮 log 未落盘（N8） |
| BUG-05 检查后工具 | **已修** | `_spend` 在 `final/checked/want_check` 时拒绝：L249；`confirm` 需 `checked`、`decision` 不可覆盖：L370-382 | `submit(); withdraw()` 仍会跑检查（N7） |
| BUG-06 缓存/长度 | **部分修复** | agent8b key 含 cap/practice/eval：L584-590；`paired_stats` 断言 pid 对齐：L193 | **mech 的缓存只按 model**（mech L53、L55-58、L116-128、L143），不含 `n_eval`/`dirs`/`practice`；agent8b `results.json` 未按配置、session 跳过（N5/N6） |
| BUG-07 SIGALRM | **已修（但引入 N1）** | `_Timeout(BaseException)`：L212-213；`_tool` 暂停/恢复：L220-230；`exec_code`：L417-436 | math_verify 不再取消框架计时 ✓；但 `alarm` 的整数取整导致工具调用重置计时（N1） |
| BUG-08 reset | **已修** | `Subject.reset` 校验 + `_guard`：L109-116；`Env.reset`：L280-282 | — |
| BUG-09 hidden_states | **已修** | 层范围：L337-340；空切片：`hidden_multi` L151-153 | — |
| BUG-10 预留切片 | **已修** | L578-582：`prac[n_practice:n_practice+n_reserved]` + 不相交断言 | — |
| BUG-11 评分 | **部分修复** | `last_boxed`：common L59-74；`grade`：common L77-89 | 多 box 已修 ✓；**EOS-at-cap 仍用 `ntok >= max_tokens`**（common L79）；`\boxed{}` 空内容 → 判错（可接受） |
| BUG-12 打印/RNG | **已修** | `gen_grade(..., verbose)`：L174-176；`generate` 的 verbose→stderr：common L42-56；每轮 `mx.random.seed(seed_base+turn)`：L521 | 工具内生成仍会往 stderr 打 throughput（进 queue 日志，不进模型上下文）✓ |
| BUG-13 final 回合 | **已修** | `env.final` + 提示消息 + 不计 code：L497、L504-518、L534 | final 回合仍执行任意代码（N9）；**submit 后无法 confirm 的陷阱**（N2） |

### A.1 BUG-01 的残留（需要处理）

- **验证**（CPU；队列用的 venv 是 numpy 2.5.5，事实一致）：`REFUSE` 正则要求方法名后跟（可含空白）`(`（L392-393），所以
  - `ff = np.fromfile` + `ff("…")` → **不拒绝**；`ld = np.load` + `ld("out/x.pkl", allow_pickle=True)` → **不拒绝**；`sv = np.save` 同理；
  - `np.lib.npyio.DataSource()` 存在（numpy 2.5.5 实测），`.open("http://…")` **不拒绝** → **可联网**；`.open("路径")` 可读本地文件；
  - 直接写 `np.fromfile(` / `np.load(` 会被拒（正则命中），但别名绕过一行即可。
- **后果**：系统提示里的 "file input/output are refused"（L447）不成立；模型若猜到路径，可读 `~/.cache/huggingface/.../MATH-500/test.jsonl`（**评测题与 gold**）或 `out/*.pkl`；用别名 `st = np.savetxt` 还可覆盖 `out/8b_results.json` 之类的文件。预留题本身已经拿不到（好），所以严重度从"直接作弊"降为"协议承诺不实 + 可读到 gold 数据集的潜在作弊"。
- **建议改法**：文本正则挡不住属性别名，改用 **AST 检查**（拒绝 `ast.Attribute` 的 `attr` ∈ {fromfile, load, save, savez, savetxt, loadtxt, genfromtxt, fromregex, tofile, memmap, DataSource, npyio, ctypeslib, …}，以及 `ast.ExceptHandler(type=None)` 的 bare except、`ast.Name` ∈ {eval, exec, compile, getattr, setattr, vars, dir, open, __import__}），或回到子进程 RPC（父进程持模型与数据）。同时把提示里的 "file I/O refused" 改成"仅 CPU/Python 计算，无文件与网络"，直到真的做到。

### A.2 BUG-06 的残留（mech 缓存 + 结果复用）

- mech：`P = f"{OUT}/{tag}_{n}.pkl"`（L53），`tag` 只含 model（L52）。因此 `eval_base`（L116）、`eval_edit/random/concise/reseed`（L124-128）、`cont`（L143）都**不随 `--n_eval` 变化**；`dirs`（L88）也不随 `--practice` 变化。用 `--n_eval 200` 重跑 400 的缓存：`R["base"]` 仍是 400 条，`S["n_eval"]=200`，`base_acc_with_2x_budget = (base_correct + correct)/len(ev)`（L151）**分母错**，结果 >1 或错位；`cont` 的题集也是旧的。
- agent8b：`res_path = out/8b_results.json`（L592）不含配置；`if name in results … "decision" in …: continue`（L646-647）会把上一次（旧 cap / 旧代码 / 旧 prompt）的 session 连同 eval 一起复用。**对媒体报告是硬伤**：重跑不会刷新数字，且没人会注意到。
- 建议：mech 的缓存键加 `n_eval`、`practice` 摘要、`dirs` 哈希（或 `--fresh`）；agent8b 的 results 按 `key`（已含 cap/practice/eval）+ **代码/prompt 哈希**分文件，或至少加 `--fresh`。
- 另外两处同类问题：`agent8b` 的 `eval_script_L{L}_a{A}`（L633）与 mech 的 `eval_edit_L{L}_a{A}`（mech L124）、`sweep_L{L}_a{A}`（mech L108）都**不含方向/练习集的哈希**；换 `--practice`/`dirs` 后旧结果会被复用。

### A.3 BUG-11 的残留（EOS-at-cap）

- `common.grade` 仍用 `ntok >= max_tokens` 判截断（L79）。`batch_generate` 的 `token_ids` 在"自然 EOS 恰好落在第 cap 个 token"时会等于 cap → 误判截断并计 0。概率很低，但既然要重写评分，顺手用 finish reason（`BatchResponse` 不直接给；可在 `batch_generate` 返回的文本里检查 EOS 或改 `max_tokens=cap+1` 再判）更稳。

### A.4 BUG-13 的残留（final 回合）

- `code_left -= 1 if not env.final else 0`（L534）与提示 "Nothing else will run"（L508-509）不一致：final 回合的代码仍被 `exec_code` 执行，`set_weight/weaken/reset` 仍可用（只要没 submit 过），所以模型可以在最后一轮装一个**从未测试过**的编辑。
- 更严重的是 N2（见下）。

---

## B. 新发现的 bug

| ID | 严重度 | 行号 | 一句话 |
|---|---|---|---|
| N1 | **高** | L220-230 + L417-436 | 工具调用把代码超时重置成"整数秒"，`while True: budget()` 永不超时；bare `except:` 还能吞 `_Timeout` → 会话可挂死 |
| N2 | **中** | L536-540、L551-552 | final 阶段第一次 `submit()` 后检查跑完即 break，没有 confirm 机会 → 检查通过的编辑被记为 withdraw |
| N3 | 中 | L480-488、L536-537 | `run_check` 无异常处理：检查失败 → 整个 session 抛错，当前轮 log 未落盘 |
| N4 | 中 | mech L76-87 | `_dirs` 没有空组保护：练习答案全截断/全答完 → nan 方向 → 编辑臂写入 nan 权重 |
| N5 | 中 | mech L53-58、L116-128、L143、L151 | 缓存不含 `n_eval/dirs/practice`；`base_acc_with_2x` 分母用 `len(ev)` 会错 |
| N6 | 中 | L592-594、L646-647 | `8b_results.json` 不按配置分键 + 按 decision 跳过 session → 旧结果被静默复用 |
| N7 | 低-中 | L359-368、L536-540 | `submit(); withdraw()` 仍会触发一次约 1 小时的检查 |
| N8 | 低 | L139-142、L520 | `assert_clean()` 失败直接 raise，当前轮不 dump（调试信息丢失） |
| N9 | 低 | L504-518、L534 | final 回合仍执行任意代码（可改权重），与提示不符 |
| N10 | 低 | mech L21 | mech 从 agent8b 导入 → 跑 1.7B/4B 也要求 8B checkpoint 存在（`glob(...)[0]` 会 IndexError） |
| N11 | 低 | mech L133-142 | 续写反事实可能把 base 前缀里已有的 box 当作"续写后答对"；且没有 edit@2× |
| N12 | 低 | L539 | 检查结果 JSON 里 `base_acc_finished` 可能是 `NaN`（无已答完答案时），模型看到 "NaN" |
| N13 | 低 | queue.sh L9 + agent8b L564 | `agent8b.py` 默认只跑 `session`，`script/reseed` 不跑（docstring 仍列着）；queue 日志每次覆盖 |

### N1　【高】工具调用重置代码超时 + bare except 吞超时（修复引入）

**行号**：`_tool` L220-230（`rem = signal.alarm(0)`；`finally: … if rem: signal.alarm(rem)`）；`_alarm_handler` L216-217；`exec_code` 设/清 alarm L421-436；`_Timeout(BaseException)` L212-213。

**触发**：任何"纯 Python + 工具调用"的循环，工具调用间隔 <1 s（`budget()`、`get_weight()` 都算工具）；或 `try: … except: pass`。

**后果**（CPU 实测，见附录）：
- `signal.alarm(0)` 返回的是**向上取整的整数秒**（实测：3 s 闹钟在 0.0/0.2/0.6/0.99 s 后都返回 3，1.4 s 后返回 2）。`_tool` 把 `rem` 原样重新装上，于是每次工具调用都把计时"截断"回整秒——只要调用够频繁，**计时永远停在同一个整数**。实测：2 s 超时 + 每 10 ms 调一次 `budget()`，跑了 6 s 仍未触发。
- 模型写 `while True: budget()` 就能让 `exec_code` 永不返回 → **整条 queue 挂死**（`queue.sh` 没有 watchdog，也没有 `--timeout`）。
- 就算不用工具，`_Timeout` 继承 `BaseException` 也挡不住 **bare `except:`**（实测能捕到）——`while True:\n try: pass\n except: pass` 同样可挂死。
- 系统提示 L447 说"each run may take {code_timeout}s of pure Python"与实际不符。

**建议改法**：
1. 在 `exec_code` 里设绝对截止时间 `deadline = time.monotonic() + timeout`（模块级变量），`_tool` 在 `finally` 里把工具耗时加回 `deadline`（`deadline += now - t_enter`，这样才真正"只排除前向传播"），然后按 `remaining = deadline - now` 重新装表：`remaining <= 0` 直接 `raise _Timeout`，否则 `signal.alarm(max(1, ceil(remaining)))`。
2. 用 AST 拒绝 bare `except`（`ast.ExceptHandler` with `type=None`）和 `except BaseException`。
3. 最稳：代码执行放子进程 + `SIGKILL`（同时解决 BUG-01 的 aliasing），父进程只做 RPC。

### N2　【中】final 阶段"检查完成但无法 confirm"的陷阱

**行号**：`agent_session` L536-540（检查）、L551-552（`if env.decision or (env.final and turn > 1 and log[-2].get("final_turn")): break`）。

**触发**：`env.final` 在第 k 轮被置位；模型在第 k 轮没有 submit；在 k+1 轮（最后的"额外回合"）第一次调用 `submit()`。

**后果**：`submit()` 只置 `want_check`；检查在代码块之后由框架跑（L536-537），跑完 `env.checked` 有了，但循环在 L551 看到 `log[-2].final_turn`（第 k 轮被标记）就 break → 模型没有机会 `confirm()` → `decision=None` → `"withdraw (no decision)"`。检查白跑（约 1 小时），一个通过的编辑被丢掉。

**建议改法**：不要用 `log[-2].final_turn` 这种间接判据。final 阶段用显式计数：每轮 `final_turns += 1`，只在 `env.decision` 时立即结束；若刚跑完检查（`checked` 从 None 变为非 None）则再给**恰好一轮**让模型 confirm，之后还没决定就强制 withdraw（`final_turns > 2` 硬上限）。

### N3　【中】`run_check` 失败会炸掉整个 session，且当前轮不落盘

**行号**：L480-488（无 try）；L536-537（循环里直接调用）。

**触发**：检查的 2×40×4096 生成过程中 OOM / MLX 错误 / `paired_stats` 断言失败。

**后果**：异常从 `agent_session` 抛出 → `main` 的 `for s` 循环终止（后面的 session 不跑）；`env.subj.locked=True` 且编辑未保存；当前轮的 `log.append`/`dump()` 在检查之后（L546-550），因此**这一轮的记录丢失**；`8b_results.json` 里没有该 session，重跑会重来（好），但已经花掉的时间白费。
**建议改法**：`try/except` 包住检查，失败时把 `checked` 置成 `{"error": …}` 并把 `want_check` 复位，让模型可以 `withdraw()`；同时在调用检查前先 `dump()`。

### N4　【中】mech 的方向提取没有空组保护 → nan 方向

**行号**：mech L76-87（`H[L][True]` / `H[L][False]`；`np.mean` → `u/norm`）。

**触发**：某模型在 practice 的 100 题上**全部截断**或**全部答完**（4B/8B 都可能；1.7B 目前是 38/62，没事）。

**后果**：`np.mean([])` → `nan`（RuntimeWarning），`u` 全 nan；`edit_layer` 把 nan 写进权重，`gen_grade` 输出垃圾或抛错；sweep 的 `_run`/`_edit` 有 `finally: restore_layer` 不会污染模型，但该模型的所有方向臂结果无效，且**没有任何报错**。
**建议改法**：`if not H[L][True] or not H[L][False]:` 直接报错退出（或退回"用所有答案的均值差"并在 summary 里标注）；agent8b 的 `script` 已有这种 guard（L624-625），mech 照抄即可。

### N5　【中】mech 缓存键不完整 + `base_acc_with_2x` 分母

**行号**：mech L52-58（`tag` 只有 model）、L116（`eval_base`）、L124-128（各 eval 臂）、L143（`cont`）、L151（`base_acc_with_2x`）。

**触发**：用不同的 `--n_eval`（或 `--practice`/`--dirs`）重跑已有缓存的模型。

**后果**：`R["base"]`/`R["edit"]`/`cont` 仍是旧题集；`S["n_eval"]` 却是新值；`base_acc_with_2x_budget` 的分子按旧题集求和、分母按新 `len(ev)` 除 → 数字直接错（甚至 >1）。`paired_stats` 的断言只保 base/edit 之间对齐，管不了跨配置。
**建议改法**：缓存键加 `n_eval` + `practice` 的摘要 + `dirs` 的哈希（或用 `--fresh`）；`S["n_eval"]` 从 `R["base"]` 的长度而不是 `ev` 取。

### N6　【中】`8b_results.json` 不按配置分键，session 跳过会复用旧结果

**行号**：L592-594（`res_path` 固定）、L646-647（`if name in results … continue`）；对比 L584（pkl key 已含配置）。

**触发**：改 `cap`/`practice`/`n_eval`/prompt/代码后，用同一个 `out/8b_results.json` 重跑。

**后果**：pkl 缓存会因新 key 重算，但 **results 里的 session 会被跳过**，`r["eval"]` 还是旧编辑在旧 base 上的数字 → 报告里"session2 +4.0"可能是上一版模型的。更糟的是，如果上一轮某个 session 因超时以 `withdraw (no decision)` 结束，也会被永久跳过、不再重试。
**建议改法**：results 文件按 `key` + 代码/prompt 哈希命名；或 `--fresh`；或把 `decision` 为 "withdraw (no decision)" 的 session 视为可重跑。

### N7　【低-中】`submit(); withdraw()` 仍会触发一次完整检查

**行号**：L359-368（submit 只置 `want_check`）、L378-382（withdraw 只置 decision）、L536（`if env.want_check and env.checked is None:` 不检查 decision）。

**触发**：一个代码块里先 `submit()` 再 `withdraw()`；或 `submit()` 后立刻 `withdraw()`。

**后果**：模型已经决定撤回，框架仍然花 ~1 小时跑 2×40 题的检查（时间被 `excluded` 扣掉，但 queue 墙钟被拉长），检查结果会被追加到一条已经没有下一轮去读的 feedback 里。
**建议改法**：L536 加 `and env.decision is None`；`withdraw()` 里清 `want_check`/`locked`。

### N8　【低】`assert_clean()` 失败时直接 raise，未 dump

**行号**：L139-142、L520。
**触发**：框架 bug 导致模块未恢复（正常不该发生）。
**后果**：会话崩溃、当轮状态丢失；但这是安全断言，崩溃比带编辑推理好。
**建议改法**：先 `dump()` 再 raise，或把 `bad` 与 `active` 写进 log。

### N9　【低】final 回合仍执行任意代码

**行号**：L504-518（提示 "Nothing else will run"）、L534（final 不计 code）、L535（`exec_code` 照样执行）。
**触发**：final 回合里模型调用 `set_weight`/`weaken`/`reset`（未 submit 时 `locked=False`）。
**后果**：模型可以在最后一轮装一个没测过的编辑并 submit；与提示矛盾。
**建议改法**：final 回合用一个受限命名空间（只暴露 `submit/confirm/withdraw`），或 `exec_code` 在 `env.final` 时先 AST 检查只允许这三个调用。

### N10-N13（低）

- **N10**：mech L21 `from agent8b import …` → `agent8b` 模块级 `MODEL_PATH = glob(...)[0]`（L31）；若 8B checkpoint 不存在，跑 1.7B/4B 会在 import 时 IndexError。建议把这些工具函数放到 `common.py` 或独立的 `weights.py`。
- **N11**：mech L133-142 的续写用 `last_boxed(前缀+续写)`；若 base 的截断前缀里已经有一个 `\boxed{}`（思考结束后又继续生成、并在前缀里留了 box 的情况），即使续写没产出新 box 也可能判对。建议记录"续写后是否出现新 box"，并在 summary 里区分。
- **N12**：L539 `json.dumps(env.checked)` 在 `base_acc_finished` 为 NaN 时输出 `NaN`（Python 可解析、严格 JSON 不能）；模型看到 "NaN" 不影响决策，但落盘时注意。
- **N13**：`queue.sh` 用 `agent8b.py --sessions 3`（默认 `--phases session`），所以 `script`/`reseed` 两相不会跑到 `8b_results.json`（mech 8B 有对应的臂，够用，但要知道数据在哪）。日志每次 `>` 覆盖。

---

## C. 对"拿去给量子位 / 机器之心发复现报告"而言，还缺什么（按价值排序，最多 5 条）

### C1. 多采样 + 多编辑的重复（最缺）
现在每个 arm 每题 **1 个答案**（论文评测是 490×8），每个 session 的编辑只在 **1 套种子（1000）** 上评一次；`reseed` 也只有 1 个种子。媒体会问"这是不是一个种子的运气"。建议：
- 同一编辑至少 3 套评测种子（如 1000/2000/3000），报"同一编辑跨种子的 Δ 均值 ± 区间"；
- 至少 3–5 个独立 session/编辑（queue 已跑 3 个 session，但只评一次），报"几个显著、几个撤回"；
- `reseed` 换成 5 个种子，得到噪声分布，而不是一个点。

### C2. 机制主张的正式分解 + 不确定性（"只来自截断"需要证明）
当前有 `base_trunc/edit_trunc`、`acc|finished`、`base@2×` 的续写，但：
- 没有把 Δacc 拆成 **截断通道**（base 截断→edit 不截断且对）与 **答完通道**（两边都答完但准确率变化）的逐题转移表；
- `acc|finished` 是**按治疗后状态选择**的量（edit 改变了哪些答案算"答完"），不能直接当因果分解；
- `base_acc_with_2x_budget` 是点估计、无 CI，而且**没有 edit@2×**（edit 自己的截断答案续写 4096 后会怎样？如果 edit 只是把截断"挪到"更大预算，这个数字会说清）；
- 建议：bootstrap over problems 给 Δ、`base@2×`、`edit@2×` 的 95% 区间；并列一张 2×2/3×3 的转移矩阵。

### C3. 对照的直接比较与"扰动大小匹配"
现在每个臂只和 base 比；但媒体/审稿人会问"edit 比 random 强，是不是因为 random 扰动更小？"。`u` 是单位向量，`W ← W − αuuᵀW` 的相对改变量 `‖ΔW‖/‖W‖ = α‖uᵀW‖/‖W‖` 取决于方向与 W 的对齐；随机方向的这个量通常与"截断方向"不同。建议：
- 报告每个臂的 `‖ΔW‖/‖W‖`，并把 random 的 α 调到与 edit **相同相对改变量**；
- 至少 5–10 个随机方向，报分布；
- 直接配对检验：edit vs random、edit vs reseed、edit vs concise、**agent-edit vs scripted-edit**（mech 的 `eval_edit` 与 agent8b 的 session eval 用同一个 `base_eval`、同一 400 题、同种子 1000，可以直接配对）；
- 预注册主终点 + Holm/BH 校正（现在 4 个对照臂 vs base，没有多重比较控制）。

### C4. 外部效度：第二个 held-out 集与分层
现在只有 MATH-500 的 400 题（practice/eval 按索引不相交，但同一数据集内可能有近似重复；论文自己的 held-out 也有重复题的问题）。媒体报告最好加：
- 第二个 held-out：AIME 2024/2025、HMMT、OlympiadBench 等与 MATH-500 不同来源的集合（注意与 MATH-500 去重），或者至少把 MATH-500 按 `level`/`subject` 分层报；
- 说明"8B 的 +x 分是不是只在 MATH-500 的简单题上"——按 level 分组的小表就够。

### C5. 可核查的复现包 + 限制声明（媒体稿的"事实底座"）
- 代码 commit hash、模型 revision（`models--mlx-community--Qwen3-8B-8bit@48a0b75b…`）、`SYSTEM`/`SUFFIX`/`CONCISE` 全文、所有种子、逐答案 JSONL（含 `prompt_ids`/`out_ids`），以及 `out/` 里 pkl 的格式说明；
- **显眼的协议偏差表**：8B（论文 27B）、8-bit checkpoint 反量化成 bf16（论文是原生 bf16 权重）、MATH-500 400 题×1 答案（论文 DeepScaleR 练习 + 490 题×8 答案评测）、检查 40 题×1 答案（论文 256 题×2 答案）；不要把这些埋在方法段；
- **失败案例也要报**：撤回的 session、random/concise 也提升的情况、1.7B/4B 的负结果；并修掉 `8b_results.json` 的 results 复用问题（N6），让复现者重跑能得到同一张表。

---

## 附：本次审查做过的验证（全部 CPU、未加载模型）

| 检查 | 方法 | 结果 |
|---|---|---|
| `REFUSE` 能否挡文件 IO | 用正则跑 7 段候选代码 | `np.fromfile(`/`np.load(` 被拒；`ff = np.fromfile` 别名**不被拒**；`np.lib.npyio.DataSource` **不被拒** |
| numpy 能力 | `import numpy`（venv 2.5.5，CPU） | `np.fromfile/load/save` 存在；`np.DataSource` 顶层无，但 `np.lib.npyio.DataSource` 存在；`np.distutils` 已移除 |
| `last_boxed` | 10 个边界用例（多 box、`\boxed  {7}`、空 box、嵌套、`\{1,2\}`、畸形尾部） | 多 box 取最后一个 ✓；嵌套 ✓；转义括号成对时 ✓；畸形尾部回退到前一个 `\boxed` ✓ |
| `signal.alarm` 语义 | 3 s 闹钟后等 0/0.2/0.6/0.99/1.4 s 读 remaining | 返回**整数**（向上取整）：3,3,3,3,2 |
| `_tool` 暂停/恢复 | 复刻 `tool()` 包装，2 s 超时，每 10 ms 调一次工具，跑 6 s | **未超时** → N1 确认 |
| `_Timeout` 可吞 | `try: while: pass except: print(...)` | bare `except:` **能**捕获 BaseException → N1 的另一条路径 |
| 数据序列化 | `json.dumps({"acc": np.float64(0.5)}, default=str)` | 输出 `{"acc": 0.5}`（np.float64 是 float 子类）→ 上一轮担心的"数字变字符串"**不成立**，已从 B 列表移除 |
| 队列状态 | 只读 `out/queue.log`、`mech_1.7b.log`、`mech_1.7b_summary.json`、`ps` | queue 15:03 启动，正在跑 mech 1.7b（practice 0.610 / trunc 0.380，方向层 [6,14,20]）；三个模型 checkpoint 都在缓存里 |
| 未做的事 | — | 未运行模型、未下载/安装、未改任何代码或 `out/` 文件；只写了本文件 |
