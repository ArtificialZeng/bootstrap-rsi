# T8 · M3 Max 上的 MLX 批量生成吞吐

当前实验保留同一模型各臂的 **`completion_batch_size=64, prefill_batch_size=16`**。下一轮优先测 **Qwen3.5-4B 的 128/16，再测 256/16**。Qwen3-1.7B 可以测 96/16，Qwen3-4B 和现有脚本的 8B 先以 64/16 为基线。接近 4096 token 时，Qwen3 每行需读约 0.47–0.60 GB 的 KV，大批量摊薄权重读取的收益有限。相同全局 seed 换 batch 会改变题目拿到的随机数，已有结果与调参后的结果应各自使用匹配的 baseline。

以下按本地 `mlx-lm 0.32.2`、`mlx 0.32.4` 核算。`MLX-LM` 指 `../.venv/lib/python3.12/site-packages/mlx_lm/`。本次只读源码、权重文件头、配置和日志，并在 CPU 上复算 tokenizer 长度及算式。

## 1. KV、循环状态与容量

令 `P` 为提示长度，普通生成的最大上下文 `L=P+4096`，续写的最大上下文 `L=P+8192`。我用本地 `tokenizer.json` 和 chat template，按 `common.py:17–38` 的切分、提示和 thinking 设置，在 400 道留出题上复算了长度。三个 Qwen3 的 base 中位数/最大值为 **73/814**，concise 为 **87/828**。Qwen3.5 分别为 **76/857** 和 **90/871**。续写用 base 提示加原始输出 ID，表中用全留出集的最大 base 提示作保守估计。出处 `mech.py:28–29,153,182–185,197–205`。

- **全注意力。** 每层每 token 的 K/V 是 `2 × n_kv_heads × head_dim × 2 bytes`。1.7B 有 28 层、8 个 KV 头、维度 128。4B/8B 有 36 层，头数和维度相同。8bit 权重经投影后生成浮点 K/V，默认仍是 BF16。Qwen3.5 有 8 层全注意力，4 个 KV 头，维度 256。出处 `MLX-LM/models/qwen3.py:43–98`、`models/qwen3_5.py:209–218,347–348`、`models/qwen3_next.py:74–99,128–148`。
- **Qwen3.5 的 24 层线性注意力。** 每层每行的递推状态 `[32,128,128]` 为 FP32，占 2 MiB。卷积状态 `[3,8192]` 为 BF16，占 48 KiB，通道数 `8192 = 2×16×128 + 32×128`。合计 **49.125 MiB/行**，与上下文长度无关。出处 `MLX-LM/models/qwen3_5.py:91–107,145–168,180–198`、`models/gated_delta.py:480–494,634–647`。

| 模型及实际加载方式 | 常驻权重 GiB | KV 每 token + 每行固定状态 | 普通生成 MiB/行，中位 base / 最长臂 | 续写 MiB/行，最长 | 理论批上限，普通 / 续写 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B BF16 | 3.205 | 112 KiB | 456.0 / 538.6 | 985.0 | **189 / 105** |
| Qwen3-4B BF16 | 7.492 | 144 KiB | 586.3 / 692.4 | 1266.5 | **141 / 79** |
| Qwen3-8B 原生 8bit | 8.105 | 144 KiB | 586.3 / 692.4 | 1266.5 | **140 / 78** |
| Qwen3-8B 现有脚本，decoder BF16 | 14.170 | 144 KiB | 586.3 / 692.4 | 1266.5 | **131 / 73** |
| Qwen3.5-4B 文本 BF16 | 7.834 | 32 KiB + 49.125 MiB | 179.5 / 204.3 | 331.9 | **483 / 301** |

权重从本地 safetensors 文件头的 `data_offsets` 求和，再按 loader 保留的张量计算。1.7B 的共享 `lm_head` 只计一次。Qwen3.5 去掉 vision，本地快照没有 MTP 张量。现有 `mech.py` 和 `agent8b.py` 将 8B 的 **252 个 decoder 线性投影**转为 BF16，embedding 和 `lm_head` 仍是 8bit。按表序，权重字节数为 `3441149952 / 8044936192 / 8702945280 / 15214602240 / 8411504128`。出处 `MLX-LM/models/qwen3.py:176–199`、`models/qwen3_5.py:350–372,408–443`、`mech.py:32–48,76–80`、`agent8b.py:35–50,637–638`、`out/test_agent8b_v2.log:1–2`。

上限采用 `floor((108 GiB − 权重)/(每 token KV × (L_max+255) + 固定状态))`，即 128 GiB 留 20 GiB。若按十进制 `108×10^9` bytes，上表普通/续写的上限依次是 `176/98、130/73、130/72、121/67、447/279`。`BatchKVCache` 按 256 token 扩容，合批会对齐到最长历史并左填充，增长余量用 255 token，表中的逻辑 MiB/行未计该余量。这里的上限只计权重和常驻缓存，实际还需承担 prefill、cache 拼接/筛选/提取、算子工作区以及内存压力。调度器已有 prefill 行转入 decode 时没有再次限流，实际 decode 行数短时可到 `B+prefill_batch_size−1`。例如现有 8B 在普通 `B=128` 时仅权重和按最长上下文计的缓存就约 **105.2 GiB**，续写 `B=64` 约 **95.6 GiB**，都应实测峰值后才加批。出处 `MLX-LM/models/cache.py:880–936,982–1054`、`generate.py:1482–1517,1826–1864`、`utils.py:74–83`。

复算配置路径。以下是 `~/.cache/huggingface/hub/` 下的本地快照，四者也各有 `model.safetensors.index.json`。

| 模型 | `config.json` 路径与行号 |
|---|---|
| Qwen3-1.7B | `models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json:9–25` |
| Qwen3-4B | `models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/config.json:9–25` |
| Qwen3-8B-8bit | `models--mlx-community--Qwen3-8B-8bit/snapshots/48a0b75b1ae72503e21e1558d040bc227510ff06/config.json:9–33` |
| Qwen3.5-4B | `models--mlx-community--Qwen3.5-4B-MLX-bf16/snapshots/475632ded9a95863da4e4b235ab9ccbc5d3cc6bf/config.json:11–67` |

## 2. 何时转为算力受限

128 GB 的 M3 Max 对应 40 核 GPU、标称 **400 GB/s**。以下用 `β` 和 `C` 表示模型算子的有效带宽和算力，两者未知。Metal 会根据形状、设备和精度切换普通 GEMM、量化 matmul 和 SDPA 路径。出处 [Apple 规格](https://support.apple.com/en-us/117737)、[MLX Metal matmul](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp)、[量化 matmul](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp)、[SDPA](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp)。

把一次 decode step 简化为读取一遍参与计算的权重 `W`，每行读自己长度 `L` 的 KV 与状态 `K(L)`，并做 `F(L)` 次浮点运算。每个输出 token 的流量约为 `W/B + K(L)`，其中 `B` 是当步实占行数。Qwen3.5 用 `K(L) ≈ 32 KiB×L + 2×49.125 MiB`，近似计状态的读和写。8B 的 `W` 从常驻权重扣掉只按 token 索引查表的 untied embedding，原生和现有脚本分别约 8.04 和 14.55 GB。全注意力的点积和加权求和约 `4 × 全注意力层数 × 查询头数 × head_dim × L` FLOP/行，加上投影每个参与矩阵乘的参数约 2 FLOP。

投影单独看，BF16 的权重算术强度约为 `B` FLOP/byte，原生 8bit、group=64 且含 scale/bias 时约为 `1.88 B` FLOP/byte。因此若有效算力/带宽之比 `C/β` 为 40–80 FLOP/byte（**灵敏度示例**），BF16 投影约在 B40–80 转折，原生 8bit 约在 B21–43。量化路径的有效算力须另测。单 token SDPA 即使理想复用每组 K/V，也只有约 `查询头数/KV 头数 = 2–4` FLOP/byte。整个 decode step 的简化转折条件是 `F(L)/(W/B+K(L)) ≥ C/β`。例如在 `L=4096`、`B=64/128/256`，整步算术强度约为 1.7B **8/9/9**、4B **14/16/16**、8B 原生 **24/26/28**、8B 现有脚本 **21/24/27**、Qwen3.5 **25/30/34** FLOP/byte。按 40–80 的假设，投影可以先受算力约束，长上下文的 SDPA 和 GDN 仍可能受带宽约束。真实的 SDPA 读放大、不同 kernel 的有效带宽、工作区和采样成本会改变这个估计。

下表是**满批、同上下文长度、只算权重摊薄与 KV/状态流量**时的吞吐倍数，公式为 `(W/64+K)/(W/B+K)`。两列给出生成中段与接近普通 cap 的情景。

| 模型 | B64 → B128，`L=2048 / 4096` | B64 → B256，`L=2048 / 4096` | 对本实验的预期 |
|---|---:|---:|---|
| Qwen3-1.7B | 1.10× / 1.05× | 1.16× / 1.08× | 128 的稳态增益预计小，256 超普通全长预算 |
| Qwen3-4B | 1.17× / 1.09× | 1.28× / 1.15× | 128 接近峰值边界，256 超普通全长预算 |
| Qwen3-8B 原生 8bit | 1.17× / 1.09× | 1.28× / 1.15× | 量化 matmul 效率未知，256 超普通全长预算 |
| Qwen3-8B 现有脚本 | 1.27× / 1.16× | 1.48× / 1.26× | 有摊销空间，128 峰值余量太窄，256 超普通全长预算 |
| Qwen3.5-4B | 1.28× / 1.22× | 1.49× / 1.37× | 128 最值得测，256 取决于稳态增益能否抵消填批和排空 |

这些倍数用来排测试优先级。若投影已经接近有效算力上限，其单位 token 成本基本不再随 B 降低。短上下文的权重摊销更明显，例如 1.7B 在 `L=256` 的纯流量估算为 1.48×/1.94×，用短 prompt 跑几十个新 token 会高估长答收益。流水、batch shape 的 kernel 效率、整词表 top-p 排序与随机采样、实际长度分布都会影响结果，端到端吞吐也可能回落。现有 1.7B 日志中的 100 题是 321 tok/s，40 题扫参曾在 136–296 tok/s 波动。8B 的 19–22 tok/s 是 3–6 行 smoke，其中一份明确反量化 252 个投影。它们是包含 prefill 的短批或混合时长记录，无法标定 B64 的纯 decode。出处 `common.py:42–50`、`out/run.log:2,10,13,16,19`、`out/8bsmoke.log:1–4`、`out/test_agent8b_v2.log:1–2`、`MLX-LM/sample_utils.py:49–73,212–229,271–273`。

## 3. 尾部、合批和可比性

- **每次调用有一次终端排空。** `BatchGenerator` 先给活跃序列各生成一个 token，按 EOS/cap 移除完成行，空位由等待队列继续 prefill 和补入。只在等待队列耗尽后，少数长行才会长时间独跑。合并同权重的 400 题条件可以减少中间的排空，平均占用提高多少取决于条件长度分布。作为量级示例，用 1.7B 的 100 道历史练习题长度重复 4 次模拟一臂，再比较三臂分开与 1200 行 FIFO 合并；按实际 P16 调度、EOS 计一步，B64 合并少约 **6.7% decode 迭代**。这是冻结输出长度、忽略每步时长的 CPU 推演，尚无合批 wall time 数据。出处 `out/practice_base.pkl`、`out/mech_1.7b.log:2`、`MLX-LM/generate.py:1778–1889,1977–2015`。
- **本轮保持各臂独立调用。** base/concise 使用 seed 1000，reseed 使用 2000，续写使用 3000。一次 `batch_generate` 只有一个共享 sampler，拼成一次调用会丢掉这些调用边界。续写得等 base 的截断结果，约 4k 的前缀和短提示混跑也会增加 prefill 填充与 decode 历史对齐开销。edit/random 或 s8/s8rand 各自使用不同权重。出处 `mech.py:153–200`、`common.py:42–47`、`MLX-LM/generate.py:1210–1238,1415–1434,1977–1995`、`models/cache.py:1010–1045`。
- **换 batch 改变同 seed 的采样轨迹。** 每次 `common.generate` 只重置一次全局 PRNG。共享 sampler 对 `[B,V]` 逐行做 top-p、top-k 和温度采样。换 B、prefill 分块、题序，或别的行更早 EOS，都会改变本题所在的行、抽样张量形状或调用次序。MLX 的 categorical 对某些单行形状还会走不同路径。**固定 logits 且在理想算术下，每行的条件采样分布相同。实际 logits 的低位舍入、归约路径和临界 token 的候选集合可能随形状变化，有限样本的答案长度、截断率和正确率都会波动。** 即使两臂设相同 batch 和 seed，分流后也不能把每题当成使用相同随机数的严格对照。出处 `common.py:42–47`、`MLX-LM/generate.py:1415–1434,1454–1517`、`sample_utils.py:49–73,212–229,271–273`、[MLX PRNG](https://ml-explore.github.io/mlx/build/html/python/random.html)、[MLX categorical 实现](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/random.cpp#L383-L442)。
- **切换生成配置时，重跑匹配的对照。** 同一比较固定题序、checkpoint 和权重处理、B/P、步长、采样参数、seed 和软件版本。新配置使用独立输出，至少配齐 base、干预和 reseed，再比较按题配对的准确率、截断率和长度。若未来需要按题锁定随机数，可用 `(实验 seed, pid, token 位置)` 派生显式 key，另外验证数值漂移和吞吐。`mech.py` 的缓存签名记录 B，未记录 P 或步长；`agent8b.py` 复用 mechanism 的 base 缓存时未核对 B，agent 自己的生成调用仍走 `common` 默认值。`mech.py` 的 practice baseline 来自 100 题整调用，sweep 臂重跑 40 题子集；调参对照也应固定同一批题和调用方式。出处 `mech.py:84–96,108,134–146,192–195`、`agent8b.py:174–176,658–670`。

低风险开关与优先级。

| 项目 | 速度、内存及结果影响 |
|---|---|
| `prefill_batch_size` / `prefill_step_size` | 普通提示都短于 1024，当前 step=2048 通常一块就能预填完，单独改成 1024 收益预计很小。续写可先测 P8、step=512 或 1024，用更小的临时激活换更多 prefill 步。改变分组/时序后同 seed 可产生不同续写，需单独记录。出处 `MLX-LM/generate.py:1209–1238,1552–1574,1826–1887`。 |
| KV 量化 / `max_kv_size` | 本装机 `batch_generate` 的 kwargs 转给 `BatchGenerator`，它没有 `kv_bits` 参数。单序列 `generate_step` 才有量化入口，而且 `QuantizedKVCache` 缺少批合并。`max_kv_size` 给 Qwen3 换成会丢历史的旋转缓存。Qwen3.5 走自带 `make_cache`，此参数被略过。KV 量化会改注意力数值，扩展批缓存后也要另立实验。出处 `MLX-LM/generate.py:305–325,826–839,1552–1564,1977–1981`、`models/cache.py:15–40,251–337`。 |
| 复用前缀缓存 | base/reseed 的相同原始提示中位数不到 100 token，收益预计有限。续写复用有机会省去重算约 4k token，但终止时提取的 cache 已含最后一个 base token，再把该 token 当输入会重复。Qwen3 可回退一 token，Qwen3.5 的递推状态需保存前一状态或携带对应的 next logits。批封装打开 `return_prompt_caches` 会保留所有完成行的 cache，优先考虑未来单独实现有界队列和逐题校验。直接改成一次生成 8192 也会改变现有续写 seed/调度。现有 `return_prompt_caches=False` 和 `return_logprobs=False` 适合生产。出处 `MLX-LM/generate.py:1391–1446,1492–1501,1696–1706,1993–2018`、`models/cache.py:154–170,388–395,596–717`。 |

## 4. 配置建议与最小 GPU 基准

以下是**新实验**的起点，所有普通对照臂共用获选配置。斜线表示 `completion_batch_size / prefill_batch_size`。续写列供未来能独立配置的续写阶段使用，当前 `mech.py` 将同一 `--bs` 传给续写，`common.py` 固定 P16。只用当前入口时，前三个 Qwen3 模型及现有脚本的 8B 先保持 64/16，Qwen3.5 验证后可试全程 128/16。出处 `mech.py:51–54,71–73,197–200`、`common.py:42–47`。

| 模型 | 普通生成起点 | 最值得测试的变体 | 单独配置续写时的起点 |
|---|---:|---|---:|
| Qwen3-1.7B BF16 | 64/16 | 96/16，长缓存峰值合格后再测 128/16 | 64/8 |
| Qwen3-4B BF16 | 64/16 | 96/16 | 48/8 |
| Qwen3-8B 原生 8bit | 64/16 | 96/8，使用原生 8bit 基线 | 48/8 |
| Qwen3-8B 现有脚本 | 64/16 | 96/8，使用现有 BF16 decoder | 48/8 |
| Qwen3.5-4B BF16 | 128/16，先和 64/16 配对测 | 256/16，先过普通和续写的峰值检查 | 128/8 |

1. **GPU 队列空闲时，先测一个模型。** 优先做 Qwen3.5 的 `64/16 → 128/16 → 64/16`，确认稳定后再测 `256/16`。用独立临时基准，显式传 B/P，模型只加载一次，使用同样的精度、sampler 和版本。装机自带 `mlx_lm.benchmark --batch-size` 只改变输入行数，调用 `batch_generate` 时没传 decode 上限，默认 B32。出处 `MLX-LM/benchmark.py:117–153`、`generate.py:1552–1564`。
2. **先测接近 cap 的满批窗口，目标每个配置只计 4–8 步。** 取 1–2 个练习题的真实长 token 前缀，预填约 4k token 一次，留出一个待喂的真实 token，再通过 `BatchGenerator.insert(..., caches=...)` 循环引用只读前缀缓存。合并后仍形成 B 行矩形 KV，可测长历史的解码吞吐。吞吐夹具用 `stop_tokens=[]`、`max_tokens=128`，填满后才记稳态。计时开始和结束同步，每组重置 MLX 的峰值计数，记录实际 `len(responses)`、token/s、峰值内存、系统内存压力及 merge 时间。短计时完成后关闭 generator 并释放批 cache，再做下一组。目标每个模型 **2–4 分钟**，包括预填的实际用时不确定。时间紧时优先测 B64 与候选的两个 4k 窗口，时间允许时用 2k 再查。Qwen3.5 另外用真实不等长提示复查，mask 是否为 `None` 会改变 GDN packed kernel 的选择。出处 `MLX-LM/generate.py:1623–1639,1812–1889,1943–1960`、`models/cache.py:674–717,1057–1086`、`models/gated_delta.py:429–478`。
3. **追加一次短端到端校验。** 每配置使用相同的练习题序列，按固定顺序循环到 `B_max+16` 行，保留生产 sampler、seed 和 EOS，上限先设 8–16 个新 token。记录 `sum(len(ids))/wall`、prefill 时间、`generation_tokens/generation_steps`、峰值和同一输入位置的 token ID 差异。8B 可从 8 token 起步，计时紧先做长缓存窗口。这能检查填批、排空和可复现性。正式截断率和准确率仍需 4096 cap 的同配置对照。测续写内存时另用接近 8k 的缓存。候选在反复测试中稳定超过 B64 的波动且峰值有余量时才采用。同负载下端到端提升低于约 10% 可优先保留 B64。出处 `common.py:42–50`、`MLX-LM/generate_utils.py:15–53,72–85,106–121`。
