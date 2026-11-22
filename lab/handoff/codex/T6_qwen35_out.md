# T6｜Qwen3.5 同族模型可行性

**建议先用 `mlx-community/Qwen3.5-4B-MLX-bf16`，在现有 MATH-500 留出集上做原始权重与全部 24 个线性注意力层 `dt_bias + 0.5` 的两臂比较。** 本机 `mlx-lm` 0.32.2 有对应的文本推理实现。按题给的吞吐量级估算，400 题每题都生成满 4096 token 时，单臂约 **2.1–2.7 小时**，两臂约 **4.1–5.4 小时**。以下结论来自 2026-11-20 的仓库元数据、论文和本机源码静态核查，实际加载、输出和速度待实测。

## 1. 公开权重、参数量和大小

Qwen 官方公开了四个名义规模不超过 9B 的聊天版及其预训练 `-Base` 版，均无需申请访问，许可证均为 **Apache-2.0**。表中名义规模来自模型卡的 Language Model 一栏。官方检查点的实存张量还计入视觉编码器和多 token 预测（MTP）模块，因此 9B 档原仓合计约 9.65B。社区列出的八个转换仓库也均公开，模型卡标为 Apache-2.0，源权重为对应的官方聊天版。大小按 Hugging Face `blobs=true` 元数据中的 `.safetensors` 文件字节数求和，1 GB = 10⁹ 字节。 [Qwen 官方合集](https://huggingface.co/collections/Qwen/qwen35)

| 文本名义规模 | 官方聊天版 | 官方预训练版 | 官方检查点实存参数 | 社区 MLX BF16 | 社区 MLX 8bit |
|---|---|---|---:|---|---|
| 0.8B | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | [873,438,784](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B?blobs=true) | [mlx-community/Qwen3.5-0.8B-MLX-bf16](https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-bf16/tree/main) · 1.71 GB | [mlx-community/Qwen3.5-0.8B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-8bit/tree/main) · 1.00 GB |
| 2B | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | [2,274,069,824](https://huggingface.co/api/models/Qwen/Qwen3.5-2B?blobs=true) | [mlx-community/Qwen3.5-2B-MLX-bf16](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-bf16/tree/main) · 4.43 GB | [mlx-community/Qwen3.5-2B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-8bit/tree/main) · 2.66 GB |
| 4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | [4,659,865,088](https://huggingface.co/api/models/Qwen/Qwen3.5-4B?blobs=true) | [mlx-community/Qwen3.5-4B-MLX-bf16](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16/tree/main) · 9.08 GB | [mlx-community/Qwen3.5-4B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-8bit/tree/main) · 5.14 GB |
| 9B | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | [9,653,104,368](https://huggingface.co/api/models/Qwen/Qwen3.5-9B?blobs=true) | [mlx-community/Qwen3.5-9B-MLX-bf16](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-bf16/tree/main) · 18.82 GB | [mlx-community/Qwen3.5-9B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit/tree/main) · 10.43 GB |

社区另有短名 `mlx-community/Qwen3.5-{尺寸}-bf16` 和 `mlx-community/Qwen3.5-{尺寸}-8bit`，对应权重分片的大小和 LFS SHA-256 均一致。上表采用模型卡有明确许可和转换来源的仓库。`mlx-community` 的公开搜索未找到这四档 `-Base` 的 BF16 或 8bit 转换。 [4B BF16 元数据](https://huggingface.co/api/models/mlx-community/Qwen3.5-4B-MLX-bf16?blobs=true) · [4B BF16 短名元数据](https://huggingface.co/api/models/mlx-community/Qwen3.5-4B-bf16?blobs=true) · [社区 Base 检索](https://huggingface.co/api/models?author=mlx-community&search=Qwen3.5%20Base)

## 2. 本机 MLX 实现与 Session 8 的参数

以下 `mlx_lm/` 源码路径相对于 `../.venv/lib/python3.12/site-packages/`，版本见同目录的 `mlx_lm-0.32.2.dist-info/METADATA:3`。所查 4B BF16 和 8bit 配置均为 `model_type: qwen3_5`。加载器据此导入 `models/qwen3_5.py`，从 `text_config` 建立语言模型。`qwen3_5.py` 复用 `qwen3_next.py` 的全注意力和稠密 MLP，Gated DeltaNet 的状态更新由 `gated_delta.py` 实现，语言模型加载时会过滤视觉张量及 MTP。源码见 `mlx_lm/utils.py:200–227,473–519` 和 `mlx_lm/models/qwen3_5.py:17–23,350–372,396–443`。 [4B BF16 配置](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16/blob/main/config.json) · [4B 8bit 配置](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-8bit/blob/main/config.json)

| 目标 | `blk = model.model.layers[i]` 下的实际属性 |
|---|---|
| 线性注意力的 time-step bias | `blk.linear_attn.dt_bias`，每个 V 头一个元素 |
| 线性注意力的 log decay rate | `blk.linear_attn.A_log`，同样按 V 头存储 |
| 写回残差流的注意力投影 | 线性层 `blk.linear_attn.out_proj.weight`，全注意力层 `blk.self_attn.o_proj.weight` |
| 写回残差流的 MLP 投影 | 两种稠密层均为 `blk.mlp.down_proj.weight` |

官方权重键 `model.language_model.layers.<i>.linear_attn.dt_bias` 在 MLX 参数树中对应 `language_model.model.layers.<i>.linear_attn.dt_bias`。4B 是 32 层，每四层中前三层为 Gated DeltaNet，第四层为全注意力。以 `blk.is_linear` 为准，应找到 **24 个形状为 `(32,)` 的 `dt_bias`**。9B 同为 24 个 `(32,)`，0.8B 和 2B 各为 18 个 `(16,)`。源码见 `mlx_lm/models/qwen3_5.py:88–130,209–239,425–443` 和 `mlx_lm/models/qwen3_next.py:74–101,151–162`。 [4B 官方权重索引](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/model.safetensors.index.json) · [4B MLX BF16 权重索引](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16/blob/main/model.safetensors.index.json) · [各尺寸模型卡](https://huggingface.co/collections/Qwen/qwen35)

**Session 8 的操作**是保存每层已加载的 `dt_bias`，对所有元素加一次 `0.5`，按原张量 dtype 写回，评测后在 `finally` 中恢复。`A_log` 保持原值。MLX 的状态保留因子为 `g = exp(-exp(float32(A_log)) * softplus(a + dt_bias))`，所以固定其他输入时，加正偏置会减小 `g`，使旧状态衰减更快。8bit 版的索引里 `dt_bias` 也是单独的浮点参数，无需为这项编辑反量化投影。建议在评测前核对目标层数、向量形状及恢复后的张量值。源码见 `mlx_lm/models/qwen3_5.py:123–126,183–194`、`mlx_lm/models/gated_delta.py:18–20,627–647`。 [4B MLX 8bit 权重索引](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-8bit/blob/main/model.safetensors.index.json) · [论文表 13](https://arxiv.org/html/2611.21885v2#A5.T13)

## 3. 4B BF16 的运行预算与脚本改动建议

**选 4B BF16**。它与现有 Qwen3-4B 同属一个名义规模，投影方向消融可以直接编辑浮点权重。建议固定此次查到的 [4B BF16 快照 `475632d`](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16/tree/475632ded9a95863da4e4b235ab9ccbc5d3cc6bf)。约 9.08 GB 是磁盘权重分片总量，含视觉权重。以现有 `bs=64`、每序列 4096 token 粗算，4B 的 8 层 BF16 KV 约 8 GiB，24 层 FP32 递推主状态约 3 GiB，另计 prompt、卷积状态及运行时临时内存。在 128 GB 上容量估算有余量，实际峰值不确定。依据 `common.py:42–50`、`mlx_lm/models/qwen3_5.py:343–348` 和 `mlx_lm/models/gated_delta.py:634–637`。 [4B 头数与维度](https://huggingface.co/Qwen/Qwen3.5-4B#model-overview)

**每臂耗时**按现有批生成的总输出吞吐计算。用题给的 1.7B ≈ 300 tok/s、8B ≈ 100–150 tok/s，把每 token 耗时 `1 / tok/s` 按名义参数量线性内插，4B 约 **170–220 tok/s**。`400 × 4096 = 1,638,400` 个新 token 对应约 **2.1–2.7 小时/臂**，两臂约 **4.1–5.4 小时**。若实测平均每题 3205 token，单臂约 **1.6–2.1 小时**。3205 是现有 1.7B 练习集均长，仅作情景估算。混合注意力、精度、批大小和提前停止都会影响实速。队列结束后可用一批练习题测总输出 tok/s 与峰值内存，再更新预算。依据 `common.py:42–50`、`out/run.log:1–3`。

1. **加载和结果标识。** 在 `mech.py` 增加明确的模型选项和本地快照路径，先确认快照存在再交给 `load()`，缺失路径可能触发 Hugging Face 下载。将结果与缓存按 checkpoint 修订、精度、评测样本和编辑配置区分。BF16 跳过现有无条件 `dequantize_projections`，该函数在每层硬取 `self_attn`，混合架构会报错。复用 `common.py` 或 `agent8b.py` 的接口时，把模块导入阶段对旧 Qwen3 快照的 `glob(...)[0]` 延后到各自加载入口。依据 `mech.py:20–23,39–64`、`common.py:11–15,29–30`、`agent8b.py:28–50`、`mlx_lm/utils.py:265–293,688–701`。

2. **先跑固定偏置的两臂。** 把原始权重与 `dt_bias + 0.5` 接到留出集评测入口，直接复用题序、提示、评分与逐题配对统计。固定偏置无需练习集提方向或扫层，两个条件各自建立生成缓存。显式传 `enable_thinking=True`，用小样核对 `<think>`、`</think>`、EOS 与末尾 `\boxed{}` 的处理，保持当前满 4096 token 算错的评分口径。当前每批只重置一个 RNG，动态批随答完顺序变化会移动随机流。如需论文式逐题同种子，给每题独立随机流，并在两臂使用同一题的 seed。依据 `mech.py:66–130`、`common.py:33–50,77–89`、`mlx_lm/generate.py:1418–1427,1977–1990`、`handoff/protocol_spec.md:208–216`。 [4B 聊天模板](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-bf16/blob/main/chat_template.jinja)

3. **再接方向消融。** `hidden_multi` 现向每层传入同一个 causal mask，多 token 时该值是字符串，线性层需要它自己的 mask。单条无 padding 的隐藏态读取可对线性层用 `None`、全注意力层用 causal mask，保持对回答 token 的层后残差取均值。`edit_layer` 和 `restore_layer` 共用上表的投影选择，逐矩阵保留原 dtype，按当前模型重算练习回答和方向。如果同时保留方向与随机方向、简洁提示和 reseed 对照，直接用原生 `batch_generate` 管理线性层 `ArraysCache` 与全注意力层 `KVCache`。依据 `agent8b.py:149–166`、`common.py:136–153`、`mech.py:72–88,118–143`、`mlx_lm/models/base.py:45–61`、`mlx_lm/models/qwen3_5.py:158–159,280–296,343–348`。

这次评测回答的是固定权重编辑能否从 27B 迁移到 4B，以及准确率变化与截断减少的关系。论文的 **+6.6 个百分点**来自 27B 在 **490 题 × 8 答案 × 16,384 token** 上的最终评测，4096 是会话内和预留题检查的预算。本地则是 **400 题 × 1 答案 × 4096 token**。 [论文 §3.6](https://arxiv.org/html/2611.21885v2#S3.SS6) · [论文表 13](https://arxiv.org/html/2611.21885v2#A5.T13)
