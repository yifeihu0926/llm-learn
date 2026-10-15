# DS 学习手册：从 token 到系统

重编版：2026-10-14。覆盖 V4 Flash / Pro 与 V4.1 Flash。

[打开带交互动画的 HTML 阅读版](ds-handbook.html)（下载后用浏览器打开，正文与动画无需联网）。

## 目录

- [01 · 阅读地图与张量语言](#c01)
- [02 · 整体模型与 mHC](#c02)
- [03 · Attention 读取路径](#c03)
- [04 · Compressor 压缩器](#c04)
- [05 · Indexer 与 top-k](#c05)
- [06 · MoE 路由与专家](#c06)
- [07 · MegaMoE 执行流程](#c07)
- [08 · 量化与缓存预算](#c08)
- [09 · V4.1 增量与共享](#c09)
- [10 · 生成、服务与训练](#c10)
- [11 · 源码索引与维护](#c11)

<a id="c01"></a>

# 01 · 阅读地图与张量语言

本手册围绕一个问题展开：**一个 token 进入 DeepSeek 后，表示如何变化，历史如何读取，专家如何计算，结果如何变成下一个 token？** 先以 V4 Flash 建立完整路径，再用差异表解释 V4 Pro，最后读 V4.1 的增量。

## 先选一条阅读路径

| 你的目标 | 推荐顺序 | 读完应能回答 |
|---|---|---|
| 建立整体认识 | 01 → 02 → 03 → 06 | Attention、MoE 和残差各做什么？ |
| 搞懂 KV cache | 03 → 04 → 05 → 08 | 压缩什么、何时写入、谁来选择、到底占多少内存？ |
| 跟踪 MoE 代码 | 06 → 07 → 08 | `[T,K_E]` 的路由结果如何变成专家任务，再合并回 token？ |
| 比较 V4 与 V4.1 | 02 → 09 | 哪些机制继续使用，哪些状态改成跨层共享？ |
| 理解完整运行 | 10 → 11 | 训练、普通生成、草稿验证和服务调度之间怎样衔接？ |

每一章先给计算目的，再给张量形状、计算顺序、示例和源码。动画与本章使用同一组术语。小规模数值一律标为**教学示例**，不代表 checkpoint 中的真实权重。

## 统一符号：先区分 token 轴与特征轴

| 符号 | 含义 | V4 Flash 示例 |
|---|---|---|
| `B` | 同时处理的序列数，batch size | 2 |
| `S` | 本次调用处理的 token 数；不是历史总长 | prefill 可为 16，逐 token decode 为 1 |
| `T=B×S` | 展平后本次 token 总数 | 2×16=32 |
| `N` | 当前请求已处理的总 token 数 | 随生成增加 |
| `p` | token 的绝对位置，从 0 开始 | 第 16 个 token 的位置为 15 |
| `D` | 每个 token 的隐藏维度 | 4096 |
| `H` | 主注意力 query 头数 | 64 |
| `d_k` | 主注意力每头宽度、共享 KV latent 宽度 | 512 |
| `H_I / d_I` | Indexer 的头数 / 每头宽度 | 64 / 128 |
| `r` | 历史压缩的步长；每 r 个新 token 提交一个槽 | CSA 为 4，HCA 为 128 |
| `K_I` | Indexer 保留的压缩槽数量上限 | 512 |
| `E / K_E` | 路由专家总数 / 每 token 选中的专家数 | 256 / 6 |
| `I` | 一个专家的中间层宽度 | 2048 |
| `R` | mHC 残差路数 | 4 |

`[B,S,D]` 的三个方括号位置是三根轴：批次、token、特征。例如 `[2,16,4096]` 表示 2 条序列，每条本次输入 16 个 token，每个 token 用 4096 个数表示。变成 `[2,16,512]` 时，token 数仍然是 16；变成 `[2,4,512]` 才减少了历史位置数。

```text
特征投影： [B,S,4096] → [B,S,512]       改每个 token 的宽度
历史压缩： [B,S,512]  → [B,约 S/r,512]  改长期槽的数量
打包布局： [B,S,D]    → [T,D]           重排 token，不丢弃 token
低位量化： [B,S,512]  → [B,S,512]       逻辑形状不变，数值精度/物理字节改变
```

生产框架的 `T` 也可能是多个不同长度请求打包后的总 token 数，并非总能反解为统一的 `B×S`。本手册先使用未做张量并行切分的逻辑形状；每 GPU 的本地形状需再套用 TP/EP 布局。

## 四种“权重”必须分开

| 名称 | 由什么得到 | 作用轴 | 生命周期 |
|---|---|---|---|
| 投影矩阵 `W` | 模型训练得到 | 输入特征→输出特征 | 随模型长期保存 |
| 压缩 gate / α | 当前 token 经 `wgate` 投影，再对窗口归一化 | 对每个特征维，沿窗口 token 轴加权 | gate 可保存在有限压缩 state；α 通常是临时计算 |
| Indexer head 权重 | 当前 query 的输入经 `weights_proj` | 聚合多个索引 head 的分数 | 当前 query 的中间结果 |
| MoE route 权重 | Router 分数中取被选专家并归一化、缩放 | 聚合被选专家输出 | 当前 token 的计算与派发元数据 |

因此，“有一个额外权重”不等于“为所有历史 token 永久多存一份 KV”。看清生成位置、归一化轴和释放时机，才能判断内存成本。

## 本手册的证据边界

- **配置事实**：固定版本 config 中的维度、层数、压缩比。
- **参考实现**：按官方 Python 源码说明调用顺序、张量和状态；它可能用 BF16 数组模拟低位缓存。
- **部署实现**：vLLM、SGLang、DeepGEMM 等固定版本的打包格式、通信和 kernel。
- **教学示例**：为了手算缩小 token、头或专家数量；不用于性能比较。

这四种证据在对应章节分别说明。当前手册是对已有 V4 / V4.1 学习资料的重编，版本固定于已记录的源码快照；版本、链接和历史资料入口统一放在第 11 章。


---

<a id="c02"></a>

# 02 · 整体模型与一个完整 Block

先把所有部件放回一条前向路径。**Attention 让当前 token 读取历史，MoE 对每个 token 做专家变换，mHC 管理子层之间的残差信息。** 三者在同一层内协作。

## 输入到输出

```text
输入 token IDs                 [B,S]
  ↓ Embedding 查词表
单路 token 表示                [B,S,D]
  ↓ 扩展为 R=4 路残差
四路状态 X                     [B,S,4,D]
  ↓ 重复 L 个 Transformer Block
四路状态 X                     [B,S,4,D]
  ↓ 最终加权收缩 + RMSNorm
单路隐藏表示                   [B,S,D]
  ↓ 词表 Head
词表 logits                    [B,V]（参考普通生成只取本次最后位置）
  ↓ 采样/选择
下一个 token ID                [B]
```

`V=129280` 是这里的词表大小。教学训练图经常画成 `[B,S,V]`，而当前 V4 参考 `ParallelHead.get_logits` 实际取 `x[:, -1]`，普通生成输出 `[B,V]`。不能把“可定义的逻辑计算”与“当前函数实际返回值”混为一谈。

[本章逐步交互演示](ds-handbook.html#c02)

## Flash 与 Pro：共同骨架，配置不同

以下来自固定版本官方推理 config，层号从 0 开始；主干层数不包含 MTP 辅助层。

| 项目 | V4 Flash | V4 Pro |
|---|---:|---:|
| `D` / 主干层数 `L` | 4096 / 43 | 7168 / 61 |
| 主注意力头 `H` / 每头 `d_k` | 64 / 512 | 128 / 512 |
| Q 低秩宽度 | 1024 | 1536 |
| O 分组数 / 每组低秩宽度 | 8 / 1024 | 16 / 1024 |
| 专家 `E` / 激活数 `K_E` | 256 / 6 | 384 / 6 |
| 专家中间宽度 `I` / 共享专家 | 2048 / 1 | 3072 / 1 |
| MoE 路由缩放 | 1.5 | 2.5 |
| 索引头 `H_I` / 索引宽度 `d_I` | 64 / 128 | 64 / 128 |
| 索引槽上限 `K_I` | 512 | 1024 |
| 局部窗口 / 残差路数 | 128 / 4 | 128 / 4 |
| 主干 SWA-only / CSA / HCA 层数 | 2 / 21 / 20 | 0 / 30 / 31 |
| token-ID 查表路由 / 分数 top-k 路由层数 | 3 / 40 | 3 / 58 |
| 辅助 MTP 层 | 1 | 1 |

Flash 的第 0、1 层只有局部 SWA；其余偶数层用 CSA，奇数层用 HCA。Pro 的第 0、1 层是 HCA，其余偶数层 CSA、奇数层 HCA。**Pro 没有纯 SWA 层，但它的每一层仍包含局部 SWA 分支。** “层类别”与“该层包含的局部分支”要分开。

## mHC 的四路状态究竟是什么

普通残差只保存一个 `D` 维向量。mHC 保存同一 token 的 4 个 `D` 维向量，为子层提供可学习的混合路径。初始化是复制同一个 embedding，后续的混合和回写才让四路不同。四路没有预设的“语法/知识/推理”等功能标签。

每个子层先把四路收缩成一个 `D` 维输入，只执行一次 Attention 或 MoE，再将结果写回四路。因此 Flash 的残差状态是 `4×4096` 个数，Attention 入口仍为 4096 维。

对单个 token，省略 `B,S`：

```text
X                        [4,D]       原残差
pre                      [4]         入口混合系数
post                     [4]         新结果注入系数
comb                     [4,4]       原残差之间的混合矩阵

h[d] = Σ_j pre[j] × X[j,d]                          → [D]
u    = F(RMSNorm(h)), F 为 Attention 或 MoE          → [D]
X_next[i,d] = post[i] × u[d] + Σ_j comb[j,i] × X[j,d] → [4,D]
```

`comb[j,i]` 表示旧路 `j` 到新路 `i` 的贡献。它不混 token，混的是**同一个 token 的残差路**。

## 三组系数怎样产生

V4 将 `X[4,D]` 展平为 `[4D]`，归一化并经过 `[24,4D]` 的投影矩阵。24 个输出分成 `4+4+16`：

```text
z = W_mix × vec(X) / sqrt(mean(X²)+eps_norm)    [24]
pre  = sigmoid(scale_pre × z_pre  + bias_pre) + eps_hc
post = 2 × sigmoid(scale_post × z_post + bias_post)
comb = Sinkhorn(scale_comb × reshape(z_comb,[4,4]) + bias_comb)
```

这里 `pre` 不是沿四路做 softmax，不能默认四个系数之和为 1。`comb` 经 20 轮 Sinkhorn 的行列归一化，形成近似双随机矩阵。`pre/post/comb` 是当前 token 计算出的系数，`W_mix/scale/bias` 是模型参数。

一个完整 Block 的执行顺序是：

1. 用 Attention 自己的 mHC 参数收缩四路，经过 `attn_norm`。
2. 执行 Attention，读取缓存并生成本 token 的注意力结果。
3. 通过 Attention 的 `post/comb` 回写四路。
4. 用 FFN 自己的 mHC 参数再次收缩，经过 `ffn_norm`。
5. 执行 MoE，汇聚路由专家与共享专家。
6. 通过 FFN 的 `post/comb` 回写四路，交给下一层。

## 一个可以手算的回写例子

**教学示例**：把 `D` 缩成 1，原四路为 `[1,2,3,4]`，取 `pre=[0.25,0.25,0.25,0.25]`，则入口 `h=2.5`。假定子层返回 `u=10`，`post=[0.1,0.2,0.3,0.4]`，`comb` 为单位矩阵，则新四路为 `[2,4,6,8]`。子层只算了一次，四个回写结果不同。本例恰好令 pre 和为 1，仅为方便手算。

## 从本章回到代码

固定版本中的 [`Transformer.forward`、`Block.hc_pre/hc_post/forward`、`ParallelHead`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py) 给出了上述路径。[Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json) 与 [Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/config.json) 定义差异。

本章只建立 Block 外部骨架。下一章进入 Attention；第 06 章进入 MoE；V4.1 的 single-pass mHC 改动集中在第 09 章。


---

<a id="c03"></a>

# 03 · Attention：一次查询到底读取哪些 KV

目标：从一个 token 的查询出发，区分 SWA、CSA、HCA 的缓存内容、读取范围和最终注意力计算。

## 输入、输出与维度

本章以已核验的 V4 Flash 官方参考实现为准。`B` 是批次数，`S` 是本次输入的 token 数，`T=B*S`，`D=4096` 是每个 token 的隐藏宽度。主注意力有 `H=64` 个查询头，每头 `d_k=512` 维。以下形状写全局逻辑形状；多卡时查询头会分片。

| 对象 | 形状 | 每个轴表示什么 |
|---|---|---|
| Attention 输入 x | `[B,S,D]` | 批次、token、隐藏特征 |
| 低秩查询 qr | `[B,S,1024]` | 每个 token 的查询中间表示 |
| 主查询 Q | `[B,S,H,512]` | 每个 token 有 H 个不同查询 |
| 本次局部 KV | `[B,S,512]` | 每个 token 一条共享 KV |
| 主注意力读地址 | `[B,S,M]` | 每个 query 读取 M 个槽，含无效占位 |
| 每头结果 O | `[B,S,H,512]` | 各头读取相同地址，得到不同结果 |
| Attention 输出 | `[B,S,D]` | 输出投影后回到隐藏宽度 |

**4096 → 512 是同一个 token 的特征投影；多个 token → 一个槽才是沿序列的压缩。** 两件事发生在不同轴上。`H*512` 也不必等于 `D`：Q 与 O 的学习投影负责进入和离开注意力空间。

## 按顺序看一遍

1. 生成查询。`wq_a` 把 D 维映射到 1024 维，经归一化后，`wq_b` 生成 H 个 512 维 query；每头再归一化，并对最后 64 维施加 RoPE。
2. 生成局部 KV。独立的 `Attention.wkv` 把 x 映射为 512 维，经过归一化、RoPE 和量化模拟。这里所有查询头共享一份 KV，缓存没有 H 这一轴。
3. 建立局部读取地址。窗口 `W=128`，query 位置 p 只能读取 `max(0,p-W+1)..p`。p 从 0 开始；窗口包括当前 token。
4. 按层类型补充全局地址。CSA 用 Indexer 选择压缩槽；HCA 列出全部已完成压缩槽。两者如何产生槽，见下一章。
5. 更新主 KV，执行稀疏注意力。根据地址取出局部 KV 与压缩 KV，计算各头分数及加权和，之后逆 RoPE、分组低秩输出投影，得到 `[B,S,D]`。

| 层类型 | 全局压缩方式 | 一个 query 实际读取的内容 |
|---|---|---|
| SWA | 无压缩器，r=0 | 因果局部窗口 |
| CSA | r=4，重叠窗口每 4 token 产一槽 | 局部窗口 + Indexer 选出的压缩槽 |
| HCA | r=128，每 128 token 产一槽 | 局部窗口 + 全部已完成压缩槽 |

这里的“全部”限定为**这一层、当前位置已经完成的压缩槽**。HCA 没有内容打分式 top-k，仍然有因果可见性限制。CSA 的 `K_I=512` 表示最多选 512 个压缩槽，不是 512 个原始 token，也不是每头分别选 512 个。

本参考实现把同一份 512 维表示同时用于打分中的 K 和加权求和中的 V。对某个 query、主注意力头 h，计算可简写为：

```text
z[h,j] = dot(Q[h], KV[address[j]]) / sqrt(512)
a[h,j] = exp(z[h,j]) / (sum_valid_j exp(z[h,j]) + exp(sink[h]))
O[h]   = sum_valid_j a[h,j] * KV[address[j]]
```

`sink[h]` 是每头可学习的额外归一化项，没有对应的 KV 值。因而所有真实 KV 权重之和可以小于 1。Indexer 的分数只负责选地址，主注意力会重新计算上面的 z。

[本章逐步交互演示](ds-handbook.html#c03)

## 教学例子：为什么地址会是 16、17

设首次 prefill 一次输入 `S=16` 个 token，位置是 0…15，并使用 CSA。prefill 的临时注意力输入排列为：

```text
临时 KV 地址： 0  1 ... 15 | 16  17  18  19
内容：       x0 x1 ...x15 | C0  C1  C2  C3
```

“16 个 prefill query”就是每个输入 token 各产生一组 query，总计 16 个位置；每个位置又有 H 个主查询头。**地址 16 是 C0 在这个临时张量中的行号，不是第 16 个历史 token。** 位置 p=7 的 query 只能使用 C0、C1，对应地址 16、17；C2、C3 虽已随整批 prefill 计算出来，仍然对 p=7 不可见。

进入逐 token decode 后，主长期缓存布局换成固定的 128 槽环形局部区加压缩区：

```text
长期缓存地址： 0 ... 127 | 128 129 130 131 ...
内容：        局部环形区 | C0  C1  C2  C3 ...
当前位置 p 的局部写地址 = p % 128
压缩槽 Cj 的地址         = 128 + j
```

所以同一个 C0 在首次 prefill 临时张量中可能是地址 16，在 decode 长期缓存中是地址 128；槽的语义没变，承载它的张量布局变了。

## 代码映射与参考

形状和调用顺序对应 [Attention.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L484)；地址生成对应 [get_window_topk_idxs / get_compress_topk_idxs](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L255)；共享 K/V 与 sink 对应 [sparse_attn_kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L277)。数值配置来自 [Flash inference/config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json)。这些是固定版本链接。

## 容易混淆的地方

- 局部区与压缩区不是互斥划分：一个近期 token 可以同时通过原始局部 KV 和压缩槽影响结果。
- 生成并缓存 KV 与“本次选中哪些 KV”是两件事。每个 query 的选择会变，已完成的历史槽可以复用。
- 选中的地址跨主注意力头共享；各头的 Q 不同，因此最终注意力权重和输出仍不同。
- 这里讲的是参考代码的计算语义。代码会模拟低精度数值，但主缓存张量使用 BF16，不能把它的形状直接当作已打包 FP8 的显存布局。


---

<a id="c04"></a>

# 04 · Compressor：多个 token 如何变成一个槽

目标：把投影、门控加权、重叠窗口和缓存寿命分开，明确每个 token、每个特征维度在压缩中做什么。

## 输入、输出与维度

压缩器读取 Attention 的输入 `x[B,S,D]`，不是把局部 KV 缓存重新读出来再压一次。设输出槽宽度为 d：主 KV 压缩器 `d=d_k=512`；CSA 内另一套 Indexer K 压缩器 `d=d_I=128`。它们结构相似、参数独立，分别服务正式注意力与快速评分。

| V4 的路径 | r | 每 token 的 value 投影 | 每 token 的 gate 投影 | 一个输出槽 |
|---|---:|---|---|---|
| CSA 主 KV | 4 | `[2*512]` | `[2*512]` | `[512]` |
| CSA Indexer K | 4 | `[2*128]` | `[2*128]` | `[128]` |
| HCA 主 KV | 128 | `[512]` | `[512]` | `[512]` |

CSA 多出来的 2 表示当前块与相邻块重叠时使用的两组特征，**不是两个 token，也不是两个注意力头**。对 Flash 的 CSA 主压缩器，`wkv.weight`、`wgate.weight` 各是 `[1024,4096]`，将同一个 4096 维 token 分别投影为 1024 个值和 1024 个门控分数。

## 按顺序看一遍

1. 用两个独立的可学习线性层生成 value 与 gate。gate 确实需要额外投影计算；它不是 value 的最大值，也不是 Indexer 的相关性分数。
2. 给 gate 加上块内位置参数 `ape`。CSA 的 ape 为 `[4,2*d]`，HCA 为 `[128,d]`；这个参数随模型保存。
3. 组织需要聚合的 token 窗口。HCA 使用 128 个不重叠 token；CSA 使用最多 8 个 token 的重叠窗口，每前进 4 个 token 产出一个槽。
4. **对每个特征维度，沿窗口内的 token 轴做 softmax**，得到该维度各 token 的权重，再逐维加权求和。
5. 对聚合后的 d 维槽做 RMSNorm、RoPE 和量化模拟，并把结果写入相应长期缓存。

令 u 是窗口内 token 的编号，a 是特征维度；P 表示相应的位置参数。核心运算是：

```text
V[u,a]     = value_projection(x[u])[a]
G[u,a]     = gate_projection(x[u])[a] + P[u,a]
weight[u,a]= exp(G[u,a]) / sum_token_v exp(G[v,a])
C[a]       = sum_token_u weight[u,a] * V[u,a]
```

每个 a 有自己的 token 权重分布。softmax 不跨特征维，也不跨注意力头。得到的 C 还会进入第 5 步，所以公式中的加权和不一定就是最终写入缓存的数值。

[本章逐步交互演示](ds-handbook.html#c04)

## 教学例子：不是 max pooling

为了把数值算清，暂把窗口缩成 2 个 token、每个 token 只有 2 个维度；这里展示池化公式，不代表模型实际配置。以下 G 已包含位置参数：

| token | V 的维度 a=0 | V 的维度 a=1 | G 的维度 a=0 | G 的维度 a=1 |
|---|---:|---:|---:|---:|
| u=0 | 1 | 3 | 0 | ln(3) |
| u=1 | 4 | 2 | ln(3) | 0 |

```text
维度 a=0 的 token 权重 = [0.25,0.75]
维度 a=1 的 token 权重 = [0.75,0.25]
C[0] = 0.25*1 + 0.75*4 = 3.25
C[1] = 0.75*3 + 0.25*2 = 2.75
```

若用 max pooling 会得到 `[4,3]`，而这里是 `[3.25,2.75]`。同一个 token 可以在某个维度贡献较大、在另一个维度贡献较小；没有为整个 token 固定一个融合比例。

## CSA 的重叠究竟如何组织

把每个 token 的 2d 维 value 明确写成 `[V前半,V后半]`，gate 同样分成两半。槽 Cj 使用“上一块 4 个 token 的前半”与“当前块 4 个 token 的后半”，拼成 `[8,d]` 后池化：

```text
C0：缺失上一块用 gate=-inf 屏蔽 + token 0..3 的后半；p=3 完成
C1：token 0..3 的前半 + token 4..7 的后半；p=7 完成
C2：token 4..7 的前半 + token 8..11 的后半；p=11 完成
C3：token 8..11 的前半 + token 12..15 的后半；p=15 完成
```

所以 r=4 是**输出步长**，稳定阶段的聚合范围是 8 个 token。第一槽只有 4 个真实 token。它们的 d 维输出不再保留前半/后半两组；这解释了为什么 CSA 中间是 1024 维，压缩槽仍是 512 维。RoPE 使用当前 4-token 块的起点，即 Cj 的位置锚点 `4*j`。

HCA 没有这组双分支：C0 聚合 token 0…127，p=127 才完成；C1 聚合 128…255，p=255 完成。统一地，在零起始位置 p 处理完后，已完成槽数为 `floor((p+1)/r)`。

## gate 会不会越存越大

需要分清三种对象：模型参数、有限工作状态、长期历史缓存。`wgate` 与 ape 是模型参数；运行中算出的 gate 分数是激活，不是另一份模型参数。

| 状态 | CSA 主压缩器 | HCA 主压缩器 | 会否随历史长度增长 |
|---|---|---|---|
| kv_state | `[B,8,1024]` FP32 | `[B,128,512]` FP32 | 否，固定窗口覆盖更新 |
| score_state | `[B,8,1024]` FP32 | `[B,128,512]` FP32 | 否，固定窗口覆盖更新 |
| 已完成主压缩槽 | `[B,已完成槽数,512]` | 同左 | 有效内容随槽数增长 |

表中写实际 B 的切片；代码按 `max_batch_size` 预分配。CSA Indexer 另有较窄的 `[B,8,256]` value/gate 工作状态，以及 `[B,已完成槽数,128]` 的长期索引 K。

因此，**gate 分数会短暂保存在压缩器的 score_state；不会为每个历史 token 永久附加到主 KV cache。** score_state 保存加过 ape 的 softmax 前分数；归一化权重在压缩时求出。CSA 保留上一块供下一次重叠使用，后续覆盖；HCA 复用 128 槽工作区。prefill 的整批投影还会产生临时张量，峰值内存不能只看长期缓存。

## 代码映射与参考

参数与状态定义见 [Compressor.__init__](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L283)，重叠拼接见 [overlap_transform](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L307)，prefill/decode、softmax 与写缓存见 [Compressor.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L316)。本章形状由该固定版本代码直接核对。

## 容易混淆的地方

- Compressor 同时出现在 CSA 和 HCA；仅 CSA 额外配备 Indexer。
- 本层局部 `Attention.wkv`、主 `Compressor.wkv`、Indexer 的 `Compressor.wkv` 是三套不同投影。
- 固定窗口的 FP32 gate/value 状态确实有开销；“不长期保存 gate”不等于压缩过程没有额外内存。
- `persistent=False` 表示这些运行状态不写进模型的 state_dict，不表示推理时它们不会驻留或更新。


---

<a id="c05"></a>

# 05 · Indexer：先合并各头分数，再选 top-k

目标：从可学习映射一路算到选中的压缩槽地址，分清索引头权重、槽分数、因果屏蔽和主注意力权重。

## 输入、输出与维度

Indexer 是 CSA 的筛选器。输入是 `x[B,S,D]` 和主查询低秩中间表示 `qr[B,S,1024]`。V4 Flash 使用 `H_I=64` 个索引查询头，每头 `d_I=128` 维，最多选择 `K_I=512` 个压缩槽。主注意力头 H 与索引头 H_I 是不同概念，即使 Flash 配置中二者都是 64。

设本次结束位置为 `end=start_pos+S`，截至本次结束可建立的索引槽数 `C_total=floor(end/r)`，CSA 的 r=4。

| 对象 | 形状 | 用途 |
|---|---|---|
| 索引 Q_I | `[B,S,H_I,128]` | 各 query 对历史的检索需求 |
| 索引 K_I_cache | `[B,C_total,128]` | 已完成槽的评分特征，各索引头共享 |
| 索引头系数 alpha | `[B,S,H_I]` | 当前 token 对不同索引头的合并系数 |
| 每头每槽点积 | `[B,S,H_I,C_total]` | 每个头对每个候选槽的匹配结果 |
| 合并后的槽分数 | `[B,S,C_total]` | 对同一个槽汇总所有索引头 |
| 输出地址 | `[B,S,min(K_I,C_total)]` | 每个 query 一组地址，无头轴 |

为避免符号混淆，本章把索引特征缓存写作 `K_I_cache`；其中 `K_I=512` 单独出现时始终表示最大选择数量。

## 按顺序看一遍

1. **生成索引 query。** Indexer 自己的 `wq_b` 将 qr 投影为 `H_I*128` 维，拆成索引头，施加 RoPE、Hadamard 旋转和 FP4 数值模拟。
2. **更新索引 K。** Indexer 内独立的 Compressor 从 x 生成 128 维压缩槽。它与主 KV 的槽 Cj 在序列位置上对应，但不是把 512 维主 KV 截取前 128 维。
3. **生成各头系数。** 独立的可学习矩阵 `weights_proj` 从当前 token 的 x 生成 H_I 个数，并乘固定缩放。每个 query 一套系数，对所有候选槽广播使用。
4. **每头评分，跨头合并。** query 与所有索引 K 点积，点积先经过 ReLU，再乘对应头系数，最后沿头轴相加。多卡分片时还会归约各卡的贡献。
5. **屏蔽未来槽，再做 top-k。** 只允许已完成槽参与选择。输出槽编号加上主 KV 区域的 offset，得到真正的主注意力读取地址。

```text
alpha[b,s,h] = weights_proj(x[b,s])[h] / sqrt(d_I * H_I)
u[b,s,h,j]   = max(0, dot(Q_I[b,s,h], K_I_cache[b,j]))
score[b,s,j] = sum_h alpha[b,s,h] * u[b,s,h,j]
可见槽数 C(p)= floor((p+1)/r)，其中 p=start_pos+s
若 j >= C(p)：score[b,s,j] = -inf
selected[b,s]= topk(score[b,s,:], min(K_I,C_total)) 的下标
```

这里的 alpha **没有 softmax，也不保证非负或和为 1**。0.6、0.4 只是便于演示的系数；真实值由当前 token 的隐藏状态、已训练的投影参数和缩放共同决定。它们不是让一个 token “二选一”的标签，也不是固定模型常数。

[本章逐步交互演示](ds-handbook.html#c05)

## 教学例子：0.2、0.7 从哪里来

把索引空间暂缩成二维、两个头，设缩放后的 alpha 是 `[0.6,0.4]`；为了只展示打分，下面向量视为已经做完前述 query/K 变换。

```text
当前 token 的 q_头0 = [1,0]，q_头1 = [0,1]
候选槽 C0 的 k     = [0.2,0.7]
候选槽 C1 的 k     = [0.8,0.1]
```

| 候选槽 | 头0：ReLU(dot) | 头1：ReLU(dot) | 合并槽分数 |
|---|---:|---:|---:|
| C0 | 1×0.2+0×0.7=0.2 | 0×0.2+1×0.7=0.7 | 0.6×0.2+0.4×0.7=0.40 |
| C1 | 0.8 | 0.1 | 0.6×0.8+0.4×0.1=0.52 |

若教学设置只选 1 槽，就选择 C1。头1 单独偏好 C0，但**不会由头1再单独产生一份 top-k**：两个头的意见已先合成一个槽分数。真实模型用 128 维向量点积，即 128 项乘积之和；这些特征来自学习投影与压缩，而不是人工指定的 0.2、0.7。

所有主注意力头接收同一组 C1 地址，然后各自用 512 维主 query 与 C1 的主 KV 重新计算正式 attention。不能把 Indexer 的 0.52 直接拿来作为 C1 的最终注意力权重。

## C3 没完成，为什么矩阵里还能有分数

首次 prefill 一次输入 16 个 token 时，程序可以批量生成 C0…C3，也可以一次计算所有 16 个 query 对这 4 槽的原始点积。这个**批处理张量中存在的原始数值**，不等于每个 query 都获准读取它。

| query 位置 p | 已完成槽数 floor((p+1)/4) | 因果可见槽 | C3 的有效分数 |
|---:|---:|---|---|
| 7 | 2 | C0、C1 | -inf |
| 11 | 3 | C0、C1、C2 | -inf |
| 14 | 3 | C0、C1、C2 | -inf |
| 15 | 4 | C0、C1、C2、C3 | 允许参与评分 |

由于 C3 包含 token 12…15 的当前块，p=14 时还不能读取它。代码在 top-k **之前**把不可见槽分数置为负无穷；如果固定宽度 top-k 仍返回了无效位置，再把这些输出地址改成 `-1`，由注意力内核忽略。逐 token decode 则只对截至当前已完成的索引槽切片评分，不需要创建未来槽。

## 压缩在选择之前，还是之后

必须指明是哪套压缩器。**用于评分的 128 维索引 K 必须先生成，Indexer 才能评分。** 主 512 维 KV 与它独立计算；在本参考代码的一次 Attention 调用里，实际顺序为：

```text
局部 KV 与主 Q → Indexer 生成索引 K → 选槽地址
             → 主 Compressor 生成/更新 512 维 KV → sparse_attn 读取
```

因此不能把“先有索引压缩特征”泛化为“主压缩 KV 在代码中一定早于 top-k 生成”。只要正式读取时，相应槽的主 KV 已完成即可。未被本 query 选中的历史槽也会保留，供后面的 query 选择；不会因本次没选中而删掉。

## 代码映射与参考

可学习矩阵与独立缓存见 [Indexer.__init__](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L384)，完整打分、屏蔽与 top-k 见 [Indexer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L402)，相对于主压缩器的执行顺序见 [Attention.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L508)。以上链接固定到核验的代码版本。

## 容易混淆的地方

- 每个 query 都能算出 H_I 个系数；一个 token 不会只能取 0.6 或 0.4 中的一个。
- `Compressor.wgate` 按“窗口 token × 特征维度”控制压缩；`Indexer.weights_proj` 按“query token × 索引头”控制评分汇总，它们不是同一套权重。
- alpha、点积与合并分数是本次查询的中间激活；长期索引缓存保存压缩 K，而不是所有历史 query 的打分表。
- HCA 不运行这个 Indexer；它使用全部因果可见的高压缩槽。


---

<a id="c06"></a>

# 06 · MoE：一个 token 怎样选择并融合专家

MoE 接收每个 token 的隐藏向量，选择少量专家分别计算，再将结果加权相加。这里解释 **V4 的模型计算规则**；下一章解释同一计算怎样交给 MegaMoE 执行。

## 先固定输入、输出和维度

设 batch 为 `B`，本次输入长度为 `S`，展开后 `T=B×S`。`D` 是隐藏宽度，`E` 是路由专家总数，`K_E` 是每个 token 选中的路由专家数，`I` 是单个专家的中间宽度。注意力头数 `H` 不出现在这个计算中：MoE 按 token 路由，并不为每个 attention head 单独选专家。

| 对象 | 形状 | 一行/一列代表什么 |
|---|---|---|
| 输入 `x` | `[T,D]` | 每行一个 token，每列一个隐藏特征 |
| Router 权重 `W_R` | `[E,D]` | 每行对应一个专家的投影参数 |
| `logits`、原始分数 `s` | `[T,E]` | 每个 token 对每个专家的分数 |
| `topk_idx`、`topk_weights` | `[T,K_E]` | 每个 token 的专家编号和对应混合系数 |
| 专家 `W_gate/W_up` | 各 `[I,D]` | 从隐藏向量投影到专家中间特征 |
| 专家 `W_down` | `[D,I]` | 将中间特征投回隐藏宽度 |
| MoE 输出 `y` | `[T,D]` | 与输入 token 一一对应，随后恢复 `[B,S,D]` |

V4 Flash 使用 `D=4096、E=256、I=2048`；Pro 使用 `D=7168、E=384、I=3072`。二者都是 `K_E=6`，另有一个每个 token 都执行的共享专家。参数来自固定版本的 [Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json) 与 [Pro 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/config.json)。

[本章逐步交互演示](ds-handbook.html#c06)

## 按顺序完成五步计算

**第一步：给所有专家算原始分数。** Router 有自己的可学习投影，不使用 KV Cache，也不使用 Indexer 的分数。

```text
z = x @ W_Rᵀ                         [T,E]
s = sqrt(softplus(z))                [T,E]
softplus(z) = log(1 + exp(z))
```

实际代码使用数值稳定的 `F.softplus`。例如 `z=0` 时 `s≈0.833`，`z=1` 时 `s≈1.146`。这里是逐元素变换；没有像 softmax 那样先让全部 `E` 个专家分数之和等于 1。

**第二步：确定专家编号。** V4 有两条明确分开的路径。

| 路由模式 | 专家编号怎样得到 | 隐藏向量的上下文是否影响编号 |
|---|---|---|
| 前三层 hash 路由 | `idx = tid2eid[token_ids]` | 同一层、同一 token ID 查到固定编号 |
| 后续分数路由 | `idx = topk(s + bias, K_E)` | 会，`s` 来自当前隐藏向量 |

`tid2eid` 是 `[词表大小,K_E]` 的整数表。token ID 是行号，该行存六个专家编号；不是用 token 在句子里的位置查表，也不是再次进行 top-k。每层拥有自己的表，因此相同 ID 不意味着跨层也选同一组专家。Flash 主干有 3 层查表、40 层分数路由；Pro 有 3 层查表、58 层分数路由，这里不把额外 MTP 层计入主干层数。

分数路由的 `bias` 是 `[E]`，同层所有 token 共用。它来自训练期的专家负载平衡：较忙专家的偏置下降，较闲专家的偏置上升。原方法从零初始化，并按历史批次负载更新；V4 报告给出的更新速度是 `0.001`。**公开推理前向读取 checkpoint 中的固定偏置，不根据你当前请求的 token 重新更新。** 参考实现将它存为 `Parameter`，这不能说明推理时会训练它。来源：[负载平衡方法](https://arxiv.org/html/2408.15664v1#S3)、[V4 训练设置](https://arxiv.org/html/2606.19348#S4.SS2.SSS2)。

**第三步：计算被选专家的混合权重。** 两种路由确定编号以后，都从不含 bias 的 `s` 中取值。

```text
selected_s = gather(s, idx)                         [T,K_E]
w = selected_s / sum(selected_s, axis=专家槽)
w = w * route_scale                               [T,K_E]
```

分母只加该 token 的六个被选项。Flash 的 `route_scale=1.5`，Pro 为 `2.5`，所以最终每行权重之和分别是 **1.5 或 2.5**。查表路由虽然编号固定，混合权重仍随当前隐藏向量变化。以上三步对应 [`Gate.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L564)。

**第四步：每个被选专家执行 SwiGLU。** 对专家收到的 `T_e` 个 token，它把 `[T_e,D]` 变成两个 `[T_e,I]`，再投回 `[T_e,D]`。参考代码的先后顺序为：

```python
gate = self.w1(x).float()       # [T_e,I]
up = self.w3(x).float()         # [T_e,I]
if self.swiglu_limit > 0:
    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
    gate = torch.clamp(gate, max=self.swiglu_limit)
a = F.silu(gate) * up           # SiLU(g)=g*sigmoid(g)
if weights is not None:
    a = weights * a             # [T_e,1] 广播到 I 个特征
out = self.w2(a.to(dtype))      # [T_e,D]
```

V4 的限幅值是 10：`up` 截到 `[−10,10]`，`gate` 只截上界，负值不截到 −10。这里的 SwiGLU gate 和决定专家去向的 Router gate 是两个不同对象。见 [`Expert.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L596)。

**第五步：按原 token 位置相加，再加入共享专家。** 用未加权的专家函数 `F_e` 表示数学关系：

```text
y_t = Σ[j=0…K_E−1] w[t,j] × F_idx[t,j](x_t) + F_shared(x_t)
```

参考实现实际上在 down 投影前乘权重。因为 down 无 bias，在理想实数运算中可移到投影后；低精度舍入和量化路径不保证逐位相同。共享专家不参与六选一类的竞争，也不乘那六个路由权重。外层的残差/mHC 更新发生在 MoE 返回之后。

## 两个小例子，分清“选谁”和“乘多少”

先用四个专家、`K_E=2` 说明选择差异。给定一个 token 的 `s=[1.146,1.458,1.304,0.833]`，`bias=[0,0,1,0]`，分数路由选择 **E2、E1**，不是 E0、E3。如果 hash 表该行恰好写着 `[E0,E3]`，就执行这两个专家；此时不能用 `s+bias` 的排名解释选择结果。

再缩小成两个 token、两个专家且 `K_E=2` 的教学模型，二者都执行 E0/E1，`route_scale=1.5`。以下是便于手算的假设值，不是模型实测。

| token | E0/E1 原始分数 | E0/E1 归一化后权重 | 乘 1.5 后的最终权重 |
|---|---|---|---|
| t0 | `[2,1]` | `[2/3,1/3]` | `[1,0.5]` |
| t1 | `[1,3]` | `[1/4,3/4]` | `[0.375,1.125]` |

若 t0 的专家输出分别为 `[2,0]`、`[0,4]`，共享输出为 `[1,1]`，则 `y0=1×[2,0]+0.5×[0,4]+[1,1]=[3,3]`。若 t1 对应输出为 `[4,0]`、`[0,2]`，共享仍为 `[1,1]`，则 `y1=[2.5,3.25]`。权重是每个 token 对每个被选专家的一个标量，广播到该专家输出的全部特征。

## 读代码时检查这四件事

[`MoE.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L629) 先展平 token，调用 `self.gate`，再用 `torch.where(indices==i)` 找到专家 i 的 token 行及专家槽，算完写回 `y[idx]`；多卡时汇总路由输出，最后加共享分支。

容易混淆的地方是：`topk_idx` 存专家编号；`topk_weights` 存系数；`E` 不等于注意力头数；这里不会创建 KV Cache。每个 token 可以有不同专家组合和不同权重，但同一 token 的整个 `D` 维向量会送往每个被选专家。V4.1 已取消前三层 hash 路由，其版本差异应在版本章阅读，不应反套到本章 V4 流程。


---

<a id="c07"></a>

# 07 · MegaMoE：把专家计算放进同一条执行流水线

上一章回答了“每个 token 选谁、怎样加权”。MegaMoE 回答的是“这些已确定的工作怎样在 GPU 上搬运、计算和汇总”。它接收 Router 已生成的编号与权重，组织专家并行执行。路由规则与执行后端要分开理解。

## 输入、输出与五步总图

沿用 `T=B×S`、隐藏宽度 `D`、专家数 `E`、每 token 路由数 `K_E=6`、专家中间宽度 `I`。多卡时本章的 `T` 指源 rank 本轮的有效 token 数，`P` 是专家并行 rank 数，均匀分片时每卡保存 `E/P` 个路由专家。attention 头数 `H` 不参与这些形状。

| 对象 | 逻辑形状 | 内容 |
|---|---|---|
| 输入 `x` | `[T,D]` | 本 rank 的原始 token 向量 |
| `topk_idx` / `topk_weights` | `[T,K_E]` | 已确定的专家编号/系数 |
| 展平分配记录 | `M=T×K_E` 条 | 每条关联一个 token 和一个专家槽 |
| 专家 e 收到的输入 | `[T_e,D]` | 来自各源 rank、路由给 e 的 token |
| 本地 L1 权重 | `[E/P,2I,D]` | 合并 gate/up 两个投影 |
| 本地 L2 权重 | `[E/P,D,I]` | down 投影 |
| 每条分配的结果 | 逻辑上 `[M,D]` | 已包含路由权重的专家贡献 |
| 最终输出 `y` | `[T,D]` | 恢复源 rank 的 token 次序 |

`M` 条记录是逻辑工作量，不代表先在某处永久存放一份完整 `[M,D]` 输入拷贝。真实实现用元数据和可复用的环形缓冲承接分块数据。

```text
① Router：[T,D] → idx、weight：[T,K_E]
          ↓ 展开成 M 条 token→expert 分配
② Dispatch：按专家所属 rank 组织索引并拉取 token
          ↓ 每个专家收到 [T_e,D]
③ GEMM1 → gate/up → SwiGLU × route weight → GEMM2
          ↓ 每条分配产生 D 维结果
④ Return + combine：送回源 token 的槽，再沿专家槽求和
          ↓ [T,D]
⑤ 加共享专家贡献 → MoE 输出 [T,D]
```

上图是依赖关系。若共享专家融合进 kernel，第⑤步的计算可提前与路由分支重叠，最终 combine 同时把共享贡献相加；它不必等第④步结束后才启动。

[本章逐步交互演示](ds-handbook.html#c07)

## 跟着一个分配记录走完全程

**① Router 先运行。** `topk_weights, topk_idx = router(x, token_ids)` 的两个返回值均为 `[T,K_E]`。对 token `t` 的第 `j` 个被选专家，记录 `(t,j,e,w)`，其中 `e=idx[t,j]`。库中的 `fp8_fp4_mega_moe` 不负责执行 Router，也不会再次选择 top-k。

**② Dispatch 保存去向，也保存回来的地址。** 专家分布在各 rank 上，记录先按 owner rank、local expert 组织。当前固定内核先交换计数与索引，专家 owner 再拉取所需的 token 向量、scale 和 route weight。来源元数据包含 `source_rank、token_idx、topk_slot`，因此两个 token 即使进入同一专家，结果也不会混淆。紧凑索引 `t×K_E+j` 可以用整除和余数恢复 t/j。见 [设备内核的 dispatch/pull](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L342)。

**③ 专家做两次 GEMM。** L1 把 gate/up 合并为一次矩阵乘法，逻辑输出 `[T_e,2I]`；SwiGLU 消去配对的一半，得到 `[T_e,I]`；L2 投回 `[T_e,D]`。实际权重行做了 gate/up 交错排列，不能把硬件内存中的前一半直接当 gate。低精度路径还携带 scale，它们不改变上述逻辑特征维度。

本固定 FP8/FP4 实现的 L1 epilogue 会将 gate/up 结果 round 到 BF16，按参数限幅，再以 FP32 计算 `SiLU(gate)×up×route_weight`，随后量化中间激活供 L2 使用。权重在这里已经乘过，后面的 combine 不再乘一次。限幅参数需要调用者传入；库的 `activation_clamp=None` 表示无穷大，不能把 V4 配置值 10 当作这个 API 的默认值。见 [L1 epilogue](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1108)。

**④ 写回原 token 的专家槽，再 combine。** 每条加权结果依据来源元数据写回源 rank 的 `(token,slot)`。同一 token 的各专家拥有各自的槽，随后由源 rank 读取这些贡献并求和，恢复 `[T,D]`。此路径不是让所有远端专家同时对同一个输出地址做 atomic add。见 [结果写回与 combine](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1281)。

**⑤ 共享专家只有一条必经分支。** 不融合时，上层另外计算 `shared(x)` 并加到 routed 输出；融合时，kernel 为每个源 token 的共享输出保留额外槽 `K_E`。这不把路由的 top-k 从六改成七；Router 仍只返回六个路由编号。最终输出不包含外层残差或 mHC 更新。

## 两个 token、两个专家的完整追踪

教学模型取 `T=2、D=2、E=2、K_E=2`，假设 E0 在 rank 0、E1 在 rank 1。这里只展示一个源 rank 发出的两行输入，省略另一源 rank 自己的输入。

```text
topk_idx     = [[0,1],       topk_weights = [[1.000,0.500],
                [1,0]]                      [1.125,0.375]]
```

展平后 `M=4`，第 j 槽的含义跟着每行的 `topk_idx` 走，**槽 0 不固定代表专家 E0**。

| 分配 | 原 token/槽 | 目标 | 已加权的 D 维结果（假设） |
|---|---|---|---|
| m0 | t0 / j0 | E0 | `1×[2,0]=[2,0]` |
| m1 | t0 / j1 | E1 | `0.5×[0,4]=[0,2]` |
| m2 | t1 / j0 | E1 | `1.125×[0,2]=[0,2.25]` |
| m3 | t1 / j1 | E0 | `0.375×[4,0]=[1.5,0]` |

Dispatch 后 E0 处理 t0、t1；E1 也处理 t0、t1。两边完成时间可以不同，但写回地址分别保留 t/j。Combine 对 t0 得到 `[2,2]`，对 t1 得到 `[1.5,2.25]`；若各自共享输出均为 `[1,1]`，最终两行就是 `[3,3]`、`[2.5,3.25]`。这些数值只说明数据归属和求和，不模拟 GPU 量化误差。

## 从清晰伪代码到融合实现

便于读懂的逻辑代码可以逐项执行：

```python
weights, indices = router(x, token_ids)       # 都是 [T,K_E]
assigned = make_assignments(indices, weights)
expert_inputs, source = dispatch(x, assigned)
contributions = experts(expert_inputs)       # 包含 route weight
y = combine(contributions, source)          # [T,D]
y += shared_expert(x)
```

真实库接口是调用者准备好输入 buffer、权重布局和输出张量，再调用 [`fp8_fp4_mega_moe`](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/mega/__init__.py#L168)。`transform_weights_for_mega_moe` 做权重/scale 布局预处理；它不执行 Router，也不把未量化权重自动变成 FP4。

固定版本的 GPU kernel 使用 persistent 调度：一批持续工作的线程分别负责拉取、装载、矩阵计算和 epilogue，反复领取分块任务。激活块就绪后，相应 GEMM 才能读取；L1 中间块就绪后，L2 才能消费；旧消费者完成后，环形槽才可复用。所谓 overlap，是把不同块的通信与计算穿插起来。动画时间轴用于展示依赖，不能当成实测耗时、固定加速比或所有 GPU 都适用的承诺。来源：[scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)、[SM100 启动路径](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp)。

## 常见误解

MegaMoE 的 scheduler 分配“哪个计算块接下来执行”，Router 决定“token 选哪个专家”，两者不能互换。`T_e` 可随专家不同，也可能为零；`T×K_E` 是分配数，不是不同 token 数。预分配容量 `T_cap` 与本轮 `T` 也不同，不能把 buffer 的所有行当作有效输入。共享专家和 CUDA shared memory 只是英文词相同，前者是模型分支，后者是片上存储。

本章描述固定 DeepGEMM 提交的源码行为；没有运行 GPU 性能测试。理解执行流程以后，再阅读量化章中的 FP4 payload、FP8 activation 与 scale 布局，才能把逻辑形状和实际存储字节数对应起来。


---

<a id="c08"></a>

# 08 · 量化、缓存与内存预算

这一章把三个容易混淆的动作拆开：**投影改变特征宽度，压缩减少历史槽数，量化减少每个数占用的位数。** 一个模型可以同时采用这三种机制。

## 先写清楚你要计算哪一种内存

| 对象 | 什么时候存在 | 随什么增长 |
|---|---|---|
| 模型权重 | 加载模型后常驻或分片驻留 | 参数规模、量化格式、并行分片 |
| 激活 / 工作区 | 前向过程中使用 | 本批 token 数、kernel 和调度方式 |
| 局部 KV 窗口 | 请求保留最近位置 | 到窗口上限后有界 |
| 主压缩 KV / Indexer K | 请求保留长期历史槽 | 长度 / 压缩步长；另看跨层共享 |
| 压缩器 state | 尚需参与后续压缩的投影值与 gate | 由窗口和重叠策略决定，有界 |
| 页表、索引与通信缓冲 | 推理服务运行时使用 | 页大小、并发、EP/TP 与后端 |

“总参数”“每 token 激活参数”“checkpoint 文件大小”“GPU 显存”是四个不同指标。只算专家的 FP4 权重大小，无法得到整模部署显存。

## V4 默认参考路径的混合精度

本表基于原版后训练 V4 Flash/Pro 参考配置：`dtype=fp8`、`expert_dtype=fp4`、`scale_fmt=ue8m0`。Base 或其他部署后端应重新读取配置。

| 对象 | 主要数值路径 | 应避免的误读 |
|---|---|---|
| 残差、普通子层输出 | BF16 承载 | FP4 模型不等于所有张量都是 FP4 |
| 一般 Linear | 动态 FP8 激活 × FP8 权重；关键累加 FP32，输出 BF16 | 输出格式不等于乘法格式 |
| 路由专家矩阵 | FP4 权重 + FP8 激活，即 W4A8 | 不能写成整段专家都只算 4 bit |
| 共享专家 | 参考构造器跟随默认 FP8 Linear | `expert_dtype` 不自动覆盖共享专家 |
| Compressor | `wkv/wgate` 与池化计算使用 FP32，输出再转换 | gate/state 的成本必须单列 |
| Router、mHC、归一化 | 关键计算使用 FP32 | 不能按大矩阵 dtype 推断它们 |
| 主 Attention 的 Q | BF16 数值路径 | 不采用 Indexer 的 FP4 Q 规则 |
| 主 KV | 448 维非 RoPE 部分量化到 FP8；64 维 RoPE 保留 BF16 | 不等于 512 维全部 FP8 |
| Indexer Q/K | 每个 128 维向量按 32 个数一组做 FP4；量化前使用 Hadamard | Indexer K 与主 KV 是不同缓存 |

分组输出投影 `wo_a` 是一个例外：checkpoint 为 FP8，但参考转换器将其还原到 BF16，前向用 BF16 einsum。FP4 参考 GEMM 也可能先解包权重到 FP8 再乘；生产后端的指令路径要查具体 kernel。

## scale 到底是什么

小位宽格式能表示的数有限。每一组用一个 scale 将原数值缩放到目标格式的有效范围：

```text
量化：   q_i = cast_Q(clip(x_i / scale, -Qmax, Qmax))
重建：   x_hat_i = q_i × scale
```

V4 的 MXFP4 使用 E2M1 数据和 UE8M0 共享 scale。E2M1 的非负有限值为 `0, 0.5, 1, 1.5, 2, 3, 4, 6`；UE8M0 scale 表示二的幂。参考 FP4 分组长度为 32：

```text
a = max(max_i(abs(x_i)), 6 × 2^-126)
scale = 2^ceil(log2(a / 6))
```

**教学例子**：组内最大幅值为 3，则 scale=0.5。原值 0.9 除以 scale 得 1.8，舍入成 FP4 可表示的 2，重建值为 1.0。shape 没变，数值发生了近似。

同一向量的所有值并不是每个都单独保存 scale。FP4 权重每行 32 个值共享 1 字节 scale：数据为 16 字节，加 scale 为 17 字节，理想载荷是 4.25 bit/value。对齐、元数据和其他精度组件还要另加。

[本章逐步交互演示](ds-handbook.html#c08)

## 量化模拟与物理打包不是同一个数

参考 kernel 的 `inplace=True` 路径会“量化→反量化”，然后将结果放回 BF16 张量。这模拟低位数值误差，但该数组仍按 BF16 分配。

| 一条缓存记录 | 参考 BF16 数组 | 一种按上述规则推导的打包载荷 |
|---|---:|---:|
| 主 KV `[512]` | 512×2 = 1024 B | 448 + 64×2 + 7 scales = 583 B |
| Indexer K `[128]` | 128×2 = 256 B | 128/2 + 4 scales = 68 B |

主 KV 如果将 scale 区域补齐到 8 字节，就变成 584 B/entry。583 是无填充的算术载荷，584 是特定补齐方式；都不等于整个分配器的每请求实测成本。V4.1 的全局缓存采用另一种 FP4 规则，见第 09 章。

## 一条请求的缓存怎样加总

定义 `C_r=floor(N/r)`。忽略未完成块、页填充和并行复制，先按“逻辑上已保留的数据”估算：

```text
SWA-only 层： local_entries × bytes_main
CSA 层：      local_entries × bytes_main
              + C_4 × bytes_main + C_4 × bytes_index
              + finite_compressor_state
HCA 层：      local_entries × bytes_main
              + C_128 × bytes_main
              + finite_compressor_state

local_entries ≤ 128
请求总缓存 = 对各层求和；B 条请求再分别求和
```

参考实现可能按 `max_seq_len` 预分配全容量，所以“当前有效记录”与“已经分配的 tensor 大小”不同。分页后端又有自己的页粒度；没有查分配方式前，不要用有效 token 数直接断言实际显存。

以 Flash 的 **BF16 参考数组** 为例，忽略有限窗口和压缩 state，长期历史的平均密度是：

```text
21 个 CSA 层 × (1024+256)/4
+20 个 HCA 层 × 1024/128
=6880 B / 原始 token / 请求
```

这个数只用于核对该参考数组口径。改成低位打包后，需同时替换主 KV 和 Indexer K 的记录大小；改成 V4.1 后，还要换成实际持有缓存的 source 层数量。

## QAT 在流程中的位置

QAT 是量化感知训练：训练前向引入部署时的低位误差，参数更新保留高精度主副本并通过相应梯度近似学习适应。V4 报告讨论路由专家权重和 Indexer Q/K 的 FP4 QAT。这不会把已有的旧高精度值“无损压缩回来”；它是训练策略。

本章的来源是固定版本 [V4 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)、[kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py) 及 [V4 技术报告](https://arxiv.org/html/2606.19348)。本手册核对的是公式、配置和实现边界，没有运行完整权重的 GPU 性能测试。


---

<a id="c09"></a>

# 09 · V4.1：从每层压缩，走向跨层共享

V4.1 Flash 的关键变化，是让多个层共用全局历史表示，并把图像、条件记忆和草稿生成接入同一主干。本章以已学过的 V4 为基线，只解释新增或改变的部分。层号从 0 开始；`B` 是批大小，`S` 是序列长度，隐藏维度 `D=5120`，四路残差为 `[B,S,4,D]`。

数值来自发布配置；计算顺序对应固定版本的官方参考代码。官方报告中的生产执行方式，会与便于阅读的参考前向有所不同。[发布配置][config] · [官方模型卡][card]

## 两个分类轴：压缩多少，与谁负责生成

V4 的 CSA/HCA 主要区分压缩比例。V4.1 的 CSA2（第二代压缩稀疏注意力）还要回答：**这一层自己生成全局 KV，还是读取前层已经生成的内容？**

| 分类轴 | 配置 | 含义 |
|---|---|---|
| 全局位置粒度 | 层 0–1：ratio 0 | 只有局部滑动窗口注意力（SWA） |
| 全局位置粒度 | 层 2–19：ratio 2 | 每两个原始 token 对应一个全局位置 |
| 全局位置粒度 | 层 20–39：ratio 1 | 每个原始 token 对应一个全局位置 |
| 跨层职责 | Full / Reindex / Reuse | 决定谁生成全局 KV、索引 K 和 top-k 位置 |

因此，ratio 2 不表示这 18 层各存一份压缩 KV；它规定共享表示的粒度。实际职责如下：

| 模式 | 层号 | 层数 | 输入 → 新生成的状态 |
|---|---|---:|---|
| SWA：局部窗口 | 0、1 | 2 | 本层隐藏态 → 本层局部 KV |
| Full：完整生成 | 2、8、14、20 | 4 | 本层隐藏态 → 全局 KV、索引 K、top-k |
| Reindex：重新选位置 | 24、28、32、36 | 4 | 本层索引 query + 共享索引 K → 新 top-k |
| Reuse：复用选择 | 其余层 | 30 | 共享全局 KV + 最近 top-k → 本层注意力结果 |

总计 `2+4+4+30=40`。例如层 24 使用层 20 的全局 KV/K，重新选择位置；层 25–27 使用层 24 的选择。**所有模式仍有本层主 query、局部 KV、注意力聚合与输出投影。** Reuse 省的是全局状态生成和索引计算。

可以把“共享”理解为多个层读取同一组已计算张量，而不是把这些层的参数合并。层 24 与层 25 的主 query 不同，即使读取相同位置，主注意力权重与输出仍可不同。这里分享的是历史内容和位置列表，不是把上一层的 attention 输出原样复制过去。

[本章逐步交互演示](ds-handbook.html#c09)

源码定位：`Attention` 判断源层，`SharedAttentionRuntime` 传递共享状态。[Attention][attention] · [共享状态容器][runtime]

## 压缩器与索引器怎样衔接

ratio 2 源层输入 `[B,S,5120]`：内容投影与 gate 投影各得到 512 维；每个维度沿两个 token 做 softmax，再加权求和、归一化，得到 `[B,floor(S/2),512]`。它没有 V4 ratio 4 的重叠窗口和块内位置参数。decode 未满两个 token 时，内容和 gate 分数留在短期 state，完成后才提交。

ratio 1 源层直接执行投影与归一化：`[B,S,5120] → [B,S,512]`，没有块内池化、gate 或未完成块 state。512 是特征维，ratio 是沿 token 轴的密度；两者不要混为一谈。

V4.1 的索引 K 由同一份**尚未加旋转位置编码的 512 维全局 latent**，通过独立 `wk` 投影和归一化得到 128 维。主 KV 与索引 K 仍是不同缓存，但不再各自运行一个压缩器。Reindex 只生成自己的索引 query，复用源层的 K。主 query 保留低秩特征归一化，移除了 V4 展开后的逐头 Q 归一化。[Compressor][compressor] · [Indexer][indexer]

索引 query 有 32 个头，每头 128 维；主注意力则有 64 个头，每头 512 维。前者决定读哪些位置，后者在这些位置重新计算内容权重。索引器的输出因此是整数位置列表，不能直接当成主注意力的 value，也不能把索引头数与主注意力头数互换。

层 20 还将历史按每 8 个位置分块，用**块内最大索引分数**选最多 2048 块，形成最多 16384 个候选位置。层 24/28/32/36 再在候选内选最多 512 个位置。这里的 max 是候选块评分，不是 KV 内容的 max pooling。最新未满块可保留，但未来 token 仍需因果掩码。参考代码先 dense 打分再 mask；只有相应稀疏后端才能少计算候选外分数。[Indexer.forward][indexer]

## CED：仍然因果的前 20 层与后 20 层

CED 是 Causal Encoder-Decoder（因果编码器—解码器）：0–19 层编码，20–39 层解码。它们共同构成因果生成主干，语言 encoder 不会读取未来 token。

输入 token 经 embedding 变成 `[B,S,D]`，扩展为四路，经 encoder 输出 `[B,S,4,D]`。层 20 先收缩四路、归一化，再投影出 decoder 共享的全局 KV。后半段各层继续计算自身注意力与专家网络，最终收缩到 `[B,S,D]`；普通生成头只返回末位置的 `[B,129280]` logits。[Transformer.forward][forward]

生产实现利用 bounded replay（有限尾部重算）：恢复近期局部状态，或只让 decoder 处理长提示词尾部。它是经过训练适配的**近似重建**。官方约 8B prefill / 16B decode 激活参数是该执行口径；参考 `forward` 仍对输入遍历全部 40 层，不能仅凭 KV 共享就推导同样省算。[技术报告 §3.2.2][report]

## Engram：从 token 片段查条件记忆

Engram 位于层 1、14 的注意力之前。输入是 token ID 历史和当前 `[B,S,4,D]` 状态。对结束于当前位置的 2/3/4-gram（连续 2/3/4 个 token）分别做多头哈希，产生 `[B,S,24]` 表地址；查出每项 256 维，共 6144 维记忆。

投影把记忆变成四路 key 和一路 value；当前残差与 key 的归一化点积经过带符号平方根和 sigmoid，生成 `[B,S,4]` gate。按 gate 把 value 写回四路，输出仍为 `[B,S,4,D]`。图像位置不参与 n-gram，也不接收 Engram 增量。

这张表存的是学习到的条件记忆，区别于 V4 MoE 的“token ID → 专家编号”表。官方约 196B 指 Engram 参数规模，不是每 token 都读取全部参数。[哈希实现][engram-hash] · [Engram.forward][engram]

## DSpark 与视觉：两个不同的入口

**DSpark 是辅助草稿分支。** 主干进入层 37、38、39 前，分别对四路残差取均值，拼成 `[B,S,15360]`，再投影成 5120 维上下文。捕获点在该 Block 的收缩与归一化之前。三个额外草稿 stage 对五个位置计算，内部包含注意力与 MoE；最后结合前一个已确定 token 的 Markov 修正顺序采样。

decode 草稿输出为 ID `[B,6]`（anchor 加五个新 token）、logits `[B,5,129280]` 和 confidence `[B,5]`。confidence 是原始标量，不自动等于接受概率。参考实现展示草稿前向，完整投机推理还需要主模型验证、接受或回退逻辑。[DSparkBlock][dspark] · [forward_spec][spec]

首次 prefill 调用主要为草稿分支预热历史窗口缓存，`forward_spec` 返回 `None`；上述输出形状针对后续 decode。五个草稿位置能够一起更新隐藏向量，但最终采样仍依赖前一个已确定 token，因而“半自回归”不等于五个互相独立的预测。

**视觉入口在语言主干之前。** 图像切成 `14×14` 像素 patch，经 32 层视觉 Transformer（ViT）得到 `[N_patch,1024]`；二维位置编码描述图像空间位置。Aligner 将 `3×3` 相邻特征拼接并投影为 `[N_image,5120]`，替换语言序列中的图像位置 embedding，再扩展成四路。视觉塔对单张图像做双向注意力，语言主干保持因果注意力。[ViT 与 Aligner][vision] · [图像合并入口][forward]

## 其余变化与 890 B/token 的边界

多路残差连接 mHC 改成 Single-Pass：输入多携带 `[B,S,4]` 的 `pre_mix`，当前子层产生的收缩系数交给下一个子层；输出是更新后的四路状态和下一份系数。最后也用传出的系数收缩。[Block.forward][block]

专家网络仍有 384 个路由专家、每 token 选 6 个，另加共享专家；移除前三层 hash 路由，全部按分数 top-k，图像位置使用专门选择偏置。输入 `[T,5120]`，router 输出编号/权重各 `[T,6]`，专家合并回 `[T,5120]`。[Gate.forward][gate]

缓存量化的关键增量是全局主 KV 改为 FP4，索引 K 也按指定 FP4 格式存储：

| 持久全局记录 | 数据与 scale | 每位置 |
|---|---|---:|
| 主 KV：512 维 | 256B FP4 + 32B scale（每 16 维一组） | 288B |
| 索引 K：128 维 | 64B FP4 + 4B scale（每 32 维一组） | 68B |

三个 ratio 2 源层和一个 ratio 1 源层，总位置密度为 `3/2+1=2.5`，因此：

```text
主 KV：288 × 2.5 = 720 B/原始 token
索引 K：68 × 2.5 = 170 B/原始 token
合计：890 B/原始 token
```

这是指定打包格式下的**全局 payload**，不含局部 SWA、压缩 state、页尾空隙与管理元数据。参考代码用 BF16 数组做量化模拟后回写，数组本身仍占 16 bit，其全局密度是 `(512+128)×2×2.5=3200 B/token`。两者对应不同存储实现；都不能直接当成整模型部署显存。[技术报告][report] · [Attention 缓存与量化][attention]

阅读本章后，应能沿一次第 24 层的计算说清四件事：它从层 20 取得什么，自己重新产生什么，向层 25 留下什么，以及哪些局部状态仍属于它自己。能追踪这四类张量，再看“压缩率”和“缓存字节”就不会把跨层共享误认为删除整层计算。

[config]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/config.json
[card]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md
[report]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
[compressor]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L429
[indexer]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L488
[attention]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L613
[runtime]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L1166
[forward]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L1241
[engram-hash]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/engram.py
[engram]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L328
[dspark]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L1100
[spec]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L1274
[vision]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/vision.py#L87
[block]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L968
[gate]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py#L792


---

<a id="c10"></a>

# 10 · 从模型前向到生成、服务与训练

前面各章说明了一个 Block 内部。本章把它放回完整系统：**前向产生 logits；生成循环决定何时继续调用前向；服务框架负责把多个请求排到设备；训练负责得到这些参数。**

## Prefill 与 decode：相同算子，不同工作规模

| 阶段 | 输入 | 主要任务 | 输出后的状态 |
|---|---|---|---|
| Prefill | 一段提示词，本次 `S` 通常大于 1 | 并行计算可因果处理的 token；构建局部 KV、压缩 state 与已完成槽 | 为后续生成准备缓存 |
| Decode | 普通逐 token 模式下 `S=1` | 当前 token 生成 Q，更新自己的 KV/state，并读取已有历史 | 有限 state 前进；块完成时追加压缩槽 |
| 草稿验证 | 一段候选 token | 主模型检查候选，按算法接受前缀并处理拒绝位置 | 只保留与已接受前缀一致的状态 |

每个 query 仍受因果约束。Prefill 的批量计算可以一次算出多个槽的候选结果，但较早 query 只能使用当时已经完成的槽。decode 也并非从头重算所有历史 KV；历史压缩槽通常已保留，当前步骤主要做新增状态和读取。

## 普通生成循环

```python
# 说明控制关系的伪代码，省略多请求调度和采样细节
logits = model(prompt_ids, start_pos=0)
next_id = sample(logits)
position = len(prompt_ids)
while not stop(next_id):
    emit(next_id)
    logits = model([next_id], start_pos=position)
    next_id = sample(logits)
    position += 1
```

温度、top-p、停止 token 和输出长度限制属于生成控制。它们影响从 logits 选哪个 token，不等于修改 Attention 中的 `K_I` 或 MoE 中的 `K_E`。

参考实现允许的 `max_seq_len` 是数组容量配置；模型声明支持长上下文，不会自动让已分配的缓存扩大。上下文窗口、输入长度、最大输出长度也应分别设置。

## V4 的 MTP 辅助路径

MTP（Multi-Token Prediction）是辅助预测模块。V4 Flash/Pro 各有一个 MTP Block，它复用主模型的 Embedding 与 Head，但有自己的融合投影与 Block 参数。

```text
输入 token ID → Embedding → RMSNorm → e_proj       [B,S,D]
主干四路状态 → RMSNorm → h_proj                     [B,S,4,D]
                                  ↓ 广播相加
                               [B,S,4,D]
                                  ↓ MTP Block
                               [B,S,4,D]
                                  ↓ MTP 收缩 / Norm / 共享 Head
                               草稿 logits
```

这不是在 43/61 个主干 Block 后总要串行再经过一个普通层。参考 `Transformer.forward` 只执行 `self.layers`；是否调用 MTP 由外围生成路径决定。V4.1 的 DSpark 使用三个额外草稿阶段，接线见第 09 章。

草稿带来的加速取决于草稿成本、主模型验证成本、接受率与批量调度。单凭“有 MTP/DSpark”无法算出固定加速倍数。接受后要提交对应 cache，拒绝后的草稿状态要回滚或作废。

## 服务框架如何调用模型

| 层次 | 要解决的问题 | 常见数据 |
|---|---|---|
| 接口与 tokenizer | 将请求文本/图像及参数转成模型输入 | token IDs、采样参数、图像特征 |
| 调度器 | 本轮让哪些请求推进多少 token | 请求状态、token 预算、prefill/decode 批次 |
| 缓存管理 | 分配、复用、回收 KV 与压缩 state | 页表、前缀映射、请求到缓存的映射 |
| Model runner | 将本轮输入交给正确设备与模型 | 打包的 `[T,D]`、位置和索引元数据 |
| 模型与 kernel | 执行 Attention、MoE、通信等 | 投影结果、索引、专家任务、输出 |
| 输出处理 | 采样、反分词、停止与流式返回 | logits、选中 token、完成标志 |

vLLM 的 V1 路径可以按前端→EngineCore/Scheduler→ModelExecutor/Worker→GPUModelRunner→模型理解；SGLang 的 SRT 路径可以按 HTTP/TokenizerManager→Scheduler→TpModelWorker/ModelRunner→模型→Detokenizer 理解。具体进程边界随部署设置变化。

Flash/Pro 配置决定模型计算的逻辑契约；后端决定分页、融合、通信和物理布局。例如同一 `[T,K_E]` 路由结果可交给不同 MoE backend，不能从 Router 的 shape 推断它一定使用 MegaMoE。

沿前向定位部署入口时，可查阅[第 11 章的 vLLM 实际算子目录](#c11-vllm-ops)：按模块列出函数、数据作用、启用条件和固定源码行号。

## 训练过程如何对应这些组件

| 阶段 | 学习/调整什么 | 与本文推理图的关系 |
|---|---|---|
| 预训练 | 广泛数据上的预测能力与模型参数 | 得到投影、专家、归一化和路由等参数 |
| SFT | 示例任务的回答方式与指令行为 | 仍训练同一模型参数 |
| RL | 在任务奖励与验证器下改进策略 | 生成 rollout，再进行参数更新 |
| OPD | 在学生自己生成的分布上向 teacher 学习 | 整合多个领域 teacher 的能力 |
| QAT | 适应低位权重/激活带来的误差 | 对应第 08 章的部署数值路径 |

V4 报告披露超过 32T token 的预训练，并描述领域 SFT/RL teacher 与多 teacher 的 on-policy distillation。V4.1 报告披露 45T token 预训练、7:1 的文本与多模态 token 混合，以及累计 34T token 时从 64K 向 1M 上下文扩展。34T 是扩展节点，不是再加 34T。

本手册不把训练报告的系统描述写成已经公开可运行的完整训练代码。`model.py` 主要帮助确认前向；数据配方、优化器与后训练细节需要查报告，不能仅从推理权重反推。

## 代码与资料入口

- [V4 固定版本 generate.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py)：普通采样与生成调用。
- [V4 固定版本 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)：`MTPBlock`、`Transformer` 与 `ParallelHead`。
- [vLLM 固定源码快照](https://github.com/vllm-project/vllm/tree/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/v1)：服务、调度与执行。
- [SGLang 固定源码快照](https://github.com/sgl-project/sglang/tree/bb9a820f09f97480bfc6d07564fb2b691b8857a3/python/sglang/srt)：SRT 请求路径。
- [V4 技术报告](https://arxiv.org/html/2606.19348) 与 [V4.1 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)：训练与后训练说明。


---

<a id="c11"></a>

# 11 · 源码索引、勘误与维护规则

这份手册是日常学习入口。模型共有机制在前八章讲一次，V4.1 的差异集中在第 09 章；旧资料作为历史备查，不再与手册平行维护多套解释。

## 每个问题去哪里找

| 问题 | 本手册 | 核心源码对象 |
|---|---|---|
| 4096 为什么变成 512？ | 01、03 | `Attention.wkv/wq_a/wq_b` |
| token 的 gate 是否要保存？ | 04、08 | `Compressor.kv_state/score_state` |
| CSA 和 HCA 的区别？ | 03、04 | `compress_ratio`、`overlap_transform` |
| 0.6/0.4、0.2/0.7、top-k 是什么？ | 05 | `weights_proj`、点积、head 求和、因果 mask |
| 前三层为什么低分专家也选中？ | 06 | `Gate.tid2eid` 与 `Gate.bias` 两分支 |
| Router 返回 tensor 是什么？ | 06、07 | `weights/indices [T,K_E]` |
| MegaMoE 怎样传输、计算、合并？ | 07 | dispatch/pull、L1/L2、combine |
| 压缩后的缓存实际占多少字节？ | 08、09 | 逻辑槽数、dtype、scale、padding、source 层 |
| V4.1 的 Reuse 跳过了什么？ | 09 | `SharedAttentionRuntime`、Attention 模式 |
| MTP/DSpark 在哪里被调用？ | 09、10 | 辅助前向与外围生成控制 |

## 固定版本的第一手源码

| 对象 | 版本 / 范围 | 入口 |
|---|---|---|
| V4 Flash | `60d8d70770c6` | [配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json) · [模型](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py) · [kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py) |
| V4 Pro | `b5968e9190ef` | [配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/config.json) · [模型](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py) |
| V4.1 Flash | `2cba9e42aa02` | [配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/config.json) · [模型](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py) · [视觉](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/vision.py) |
| DeepGEMM MegaMoE | `057ca5964aae`，SM100/NVLink 路径 | [Python 接口](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/mega/__init__.py) · [设备实现](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) |
| vLLM | `c91dccc08044b` | [固定快照](https://github.com/vllm-project/vllm/tree/c91dccc08044b1269f351ceec13070fe8f075e10) |
| SGLang | `bb9a820f09f9` | [固定快照](https://github.com/sgl-project/sglang/tree/bb9a820f09f97480bfc6d07564fb2b691b8857a3) |

技术报告入口：[V4](https://arxiv.org/html/2606.19348)、[V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)。模型卡和 `main` 下的报告链接可能更新；本手册的实现结论优先对应表中的固定源码版本。

<!-- vllm-runtime-start -->

<a id="c11-vllm-ops"></a>

## vLLM DeepSeek V4 实际算子目录

依据本地 vLLM **c91dccc08044b1269f351ceec13070fe8f075e10**，以 NVIDIA 文本主干为主，附可选路径与其他平台概览。相关源码在本次阅读时无本地修改。这里总结的是从forward追到的实际调用入口及其可达分支；没有运行GPU profiler，具体部署的一次执行子集仍由硬件、配置、shape与安装扩展决定。

### 先把算子放回模型

![vLLM DeepSeek V4 主干与Attention算子路径](assets/vllm-v4-operators-map.svg)

**读图。** 首层先把embedding作为重复C路残差的来源；mHC pre收缩并融合子层输入Norm。Attention输出经post/pre进入MoE；MoE结果携带residual/mix返回，交给下一层开头回写。最后一层独立post，然后hc_head收束、最终Norm，compute_logits再投词表。图中每个框可能包含多个kernel，也可能与相邻步骤部分融合。

记号：T是该调用实际token行数（可含padding，SP下可能是本rank shard）；D为hidden，C=hc_mult，I为expert intermediate，E为专家数，K_e为每token激活专家数，K_i为Indexer选择的历史压缩数。主Q/KV head_dim=512、RoPE尾部64；Flash/Pro使用同一实现，具体层数和宽度从config读取。

| 架构模式 | 本实现的ratio | 实际增加的工作 | 聚合入口 |
|---|---|---|---|
| SWA | 配置0映射到内部1 | 局部因果窗口索引 | 所选sparse MLA后端 |
| CSA / C4A | 4 | 主KV压缩、Indexer K压缩、Q评分、历史top-k | 同一后端，增加压缩候选 |
| HCA / C128A | 128 | 粗粒度KV压缩、枚举全部可见摘要 | 同一后端，增加粗摘要候选 |

CSA的Indexer是Attention内部步骤；MoE Router选择专家；Sampler选择词表token。这三个top-k处理不同对象。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L733)

### 按模块检索实际入口

每行按一个入口或功能组整理，复合函数可以启动多个kernel。搜索与模块按钮会筛选本节的主干、条件分支及可选路径表；它们不是“同一次前向全部运行”的launch数量。

<!-- ops-toolbar -->

#### [通用] Embedding、动态量化与普通 Linear

普通 Linear 经 quant_method.apply 进入所选实现。以下是主流 block-FP8 可达路径；ignored / unquantized 层走普通矩阵乘法。候选后端按硬件、shape、配置及可用扩展选择。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 词表查表 | `VocabParallelEmbedding.forward` / `F.embedding` / `ops.vocab_parallel_embedding` | token IDs → [T,D]；TP 时 mask 或融合查表，再归约各 shard。 | 首个 PP rank；inputs_embeds 已给定时可跳过查表。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/vocab_parallel_embedding.py#L518) |
| 动态 FP8 量化 | `QuantFP8` / `per_token_group_quant_fp8` / `per_token_group_quant_fp8_packed_for_deepgemm` | 激活 → FP8 data + scale；普通 block128 Linear 常按 K 每128值分组。 | 由所选 GEMM 决定是否外置量化，以及 scale packing。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/quantization/input_quant_fp8.py#L83) |
| DeepGEMM Linear | `fp8_gemm_nt_op` / `fp8_gemm_nt` | FP8 A[T,K] × FP8 W[N,K]，补 scale，输出 BF16 [T,N]。 | DeepGemmFp8BlockScaledMMKernel 获选；支持 dtype、shape 与设备。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py#L112) |
| 动态 FI / DG Linear | `dynamic_flashinfer_deepgemm_blockscale_gemm` | 小 T 使用 FlashInfer 内部量化；大 T 外置量化后调用 DeepGEMM。 | 该动态类获选且双方支持；通常 T<32 选 FI，batch-invariant 强制 DG。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py#L229) |
| CUTLASS Linear | `CutlassFp8BlockScaledMMKernel` / `ops.cutlass_scaled_mm` | 动态 FP8 量化后做 block-scaled GEMM。 | selector 选中该 block-FP8 实现。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L276) |
| Triton Linear | `TritonFp8BlockScaledMMKernel` / `w8a8_triton_block_scaled_mm_func` | FP8 block-scaled GEMM，输出恢复为调用层 shape。 | selector 选中 Triton fallback。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/linear/scaled_mm/triton.py#L159) |

实际选择器是 [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/linear/__init__.py#L701)。CUDA 注册表还含 B12x、Marlin、Humming、Torch 等按资格选择的 fallback；注册存在本身不能证明某次前向执行了它。

#### [mHC] 四路残差、系数生成与融合

记 C=hc_mult，X 为 [T,C,D]；系数投影输出 C(C+2) 个值，拆成 pre、post 与 C×C residual mix。NVIDIA decoder 直接使用下面的 TileLang 封装。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 首层广播 pre | `mhc_pre_broadcast_tilelang` | [T,D] → 保存 C 路 residual，并生成已 RMSNorm 的 [T,D] 子层输入。 | 首个 PP rank 首次 2D 输入；预合并广播投影权重。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L698) |
| prenorm 投影 | `tf32_hc_prenorm_gemm` / `_HC_PRENORM_GEMM_TILELANG_KERNEL` | 计算 v·Fᵀ 和平方和；用于归一化系数投影。 | 普通 pre：DG 可用用 TF32 helper，否则 TileLang；广播 pre 强制 DG helper。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L22) |
| gate / Sinkhorn / 收缩 | `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` | inverse RMS → sigmoid pre/post gates → Sinkhorn → C路加权收缩 → layer-input RMSNorm。 | pre 及 post/pre 后续步骤；decoder 传入 Norm 权重以融合归一化。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L542) |
| post/pre 复合调用 | `mhc_fused_post_pre_tilelang` / `_MHC_FUSED_TILELANG_KERNEL` | 旧子层输出回写 residual，并准备下一子层的 pre；之后仍运行 big_fuse。 | 本 rank T≤32 且 D%128=0 用融合前段；否则拆成 post + prenorm GEMM。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L814) |
| 独立 post | `mhc_post_tilelang` / `_MHC_POST_TILELANG_KERNEL` | X_new[j] = post[j]·u + Σ_i mix[i,j]·X_old[i]，输出 [T,C,D]。 | 最后一个本地 decoder；aux hidden 重建；post/pre 的拆分分支。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L791) |

mHC 中 residual 为 BF16，系数/关键统计为 FP32，prenorm 可用 TF32 GEMM。FFN 的 post 留给下一层开头；本地循环末尾才独立重建。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1320)

#### [Attention] 投影、Q/KV 准备、索引与稀疏聚合

主 latent/head_dim=512，尾部64维为 RoPE。低秩 Q/KV 的 learned RMSNorm 与投影后逐 query head 的无权 RMSNorm 是两次不同计算。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 主输入投影 | `fused_wqa_wkv` / `wq_b` | [T,D] → q_low + KV；q_low 再投影成 [T,Nq,512]。 | 每个 attention；底层由 Linear 量化方法决定。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L471) |
| 低秩 Q/KV Norm | `fused_q_kv_rmsnorm` / `FusedQKVRMSNormKernel` | 两路各自做 learned RMSNorm，保持各自 shape。 | 主 fused projection 拆分之后。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L519) |
| packed Q/KV 准备 | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | Q逐头无权Norm + RoPE；KV RoPE + 448维G64 FP8 / 64维BF16尾部，写 SWA cache。 | packed uint8 fp8_ds_mla cache；含 query head padding。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L793) |
| 完整 BF16 cache 准备 | `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert` | Q norm/RoPE，KV 以完整512维 BF16写cache。 | plain BF16 cache 分支。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L819) |
| 完整 FP8 cache 准备 | `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert` | Q/KV 按 per-tensor scale 输出 FP8；完整512维cache。 | plain per-tensor FP8 分支，区别于 packed 混合格式。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L832) |
| SWA 可见地址 | `_compute_swa_indices_and_lens_kernel` | 逻辑因果窗口 → 分页物理 slots 与有效长度；无效位置 -1。 | 各层 SWA metadata；图片跨度和 DSpark 有专用可见规则。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/v1/attention/backends/mla/sparse_swa.py#L1008) |
| HCA 压缩地址 | `build_c128a_topk_metadata` / `BuildC128ATopkMetadataKernel` | 枚举 floor((position+1)/128) 个可见压缩项。 | ratio=128；名称含 topk，但不运行 learned score top-k。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/sparse_mla.py#L229) |
| 压缩逻辑ID转地址 | `compute_global_topk_indices_and_lens` | 压缩记录 ID → compressed cache 物理slot，并统计有效长度。 | CSA decode 等地址转换；不重新按score排序。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/ops/cache_utils.py#L591) |
| prefill KV gather | `dequantize_and_gather_k_cache` / `DequantGatherKCacheKernel` | 分页 packed KV → BF16连续workspace，补回对应scale。 | FlashMLA prefill；CuTe DSL可用则CuTe，否则Triton。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/flashmla.py#L326) |
| prefill 候选合并 | `combine_topk_swa_indices` / `CombineTopkSwaIndicesKernel` | 压缩局部ID + SWA位置 → gathered workspace中的 combined indices。 | FlashMLA prefill；两类候选随后共用一次softmax。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/flashmla.py#L356) |
| FI 混合索引 | `build_flashinfer_mixed_sparse_indices` | SWA与compressed候选组成混合storage-offset索引，并处理页stride。 | FlashInfer SM100类；SM120保留两组indices。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/ops/cache_utils.py#L1286) |
| FlashMLA decode | `flash_mla_with_kvcache` | Q读取paged SWA + optional compressed cache，含sink，输出attention向量。 | FlashMLA获选，decode部分。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/flashmla.py#L248) |
| FlashMLA prefill | `flash_mla_sparse_fwd` | BF16 gathered KV + combined indices → sparse attention输出。 | FlashMLA获选，prefill部分，按至多4请求分chunk。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/flashmla.py#L390) |
| FlashInfer sparse MLA | `trtllm_batch_decode_sparse_mla_dsv4` | 读SWAcache与compressed cache、索引、sink，计算稀疏聚合。 | FI SM100或SM120；prefill/decode都调用该API，布局与参数分支不同。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py#L492) |
| 逆RoPE / 输出量化 | `fused_inv_rope_fp8_quant` / `FusedInvRopeFP8QuantKernel` | attention输出尾部逆旋转，按O分组重排；FP8 wo_a时同时量化。 | 普通O路径；FI SM100的RopeQuant epilogue满足条件时已在attention内完成。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L61) |
| 分组 O_a | `fp8_einsum` / `torch.bmm` | 每组attention向量 → BF16低秩 z[T,G,r_o]。 | FP8 wo_a用DG fp8_einsum；非FP8 wo_a用bmm。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L89) |
| O_b 输出投影 | `wo_b` / `RowParallelLinear` | flatten z后投影回[T,D]，按TP/SP配置归约。 | 每个attention的最后一步。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L106) |

稀疏聚合的数学主体为 score=Q·KV/√512，SWA与压缩候选一起归一化；learned sink增加softmax分母。SWA/CSA/HCA改变候选生成，不是把三次独立attention输出相加。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/flashmla.py#L174)

#### [Compressor] 历史状态与压缩 KV

只有 ratio>1 的层运行主 Compressor；CSA 同时为 Indexer 运行128维K Compressor。C4压缩使用相邻两块、共8个token；C128使用128个token。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 压缩输入投影 | `fused_wkv_wgate` / `torch.mm(out_dtype=torch.float32)` | hidden → FP32 KV与逐坐标score。 | 主Compressor或Indexer Compressor启用；projection可放aux stream。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L672) |
| 保存部分状态 | `_SAVE_PARTIAL_STATES_KERNEL` / `SavePartialStatesKernel` | KV与score+APE写分页FP32 state；未到压缩边界也要保存。 | Compressor每轮有效输入。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/compressor.py#L354) |
| C4主KV融合压缩 | `SparseAttnCompressNormRopeStoreC4Kernel` / `SparseAttnCompressNormRopeStoreFullC4Kernel` | 逐latent坐标softmax压缩 → RMSNorm → RoPE → cache。 | 主head512、ratio4；按packed/full cache二选一；每4token完成一项。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py#L2398) |
| C128压缩第一阶段 | `SparseAttnCompressC128Block8Kernel` | 从分页历史读取128token，逐坐标压缩到FP32 temporary。 | 主head512、ratio128；到边界才输出压缩项。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py#L2436) |
| C128第二阶段 | `SparseAttnNormRopeStoreKernel` / `SparseAttnNormRopeStoreFullKernel` | temporary → Norm / RoPE / packed或full cache。 | C128第一阶段后；非FULL graph且无边界时可提前跳过。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py#L2461) |
| Indexer K压缩 | `compress_norm_rope_store_triton` / `FusedKVCompressNormRopeInsertIndexerTritonKernel` | 128维Indexer K做C4压缩、Norm、RoPE、FP8或MXFP4量化写cache。 | CSA Indexer；默认FP8，显式mxfp4配置走FP4device分支。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/compressor.py#L470) |

压缩门控可写为：k̄[d]=Σ_j softmax_j(score[j,d]+APE[j,d])·KV[j,d]，随后Norm与块起点RoPE。每个坐标有自己的权重，不能简化成每token只有一个门控标量。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py#L711)

#### [Indexer] CSA 的评分与历史 top-k

仅 ratio=4 创建 learned Indexer。它位于 Attention 内，输出压缩位置；MoE Router 另外选择专家，词表采样则选择 token。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| Q与head权重投影 | `indexer.wq_b` / `indexer.weights_proj` | q_low→Q[T,Ni,128]；hidden→各Indexer head评分权重。 | CSA；短上下文shortcut可能跳过Q投影。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L1109) |
| Indexer Q融合准备 | `fused_indexer_q_rope_quant` / `IndexerQFp8Kernel` / `IndexerQMxFp4Kernel` | Q尾部RoPE + BF16舍入 + FP8或G32 MXFP4；输出缩放后的head权重。 | CuTe DSL可用选CuTe，否则对应Triton；FP4要求SM100 family与显式配置。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py#L547) |
| 短上下文直接全选 | `_fill_short_context_topk_indices` | 填所有已完成压缩位置，其余-1，跳过Q评分与top-k。 | 最大压缩长度≤index_topk，且不在CUDA stream capture中；K Compressor仍执行。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L1085) |
| prefill Indexer K gather | `cp_gather_indexer_k_quant_cache` | 分页量化K与scale → 连续chunk工作区，保留FP8/FP4表示。 | Indexer prefill评分前；不是主KV反量化gather。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/sparse_attn_indexer.py#L484) |
| prefill Indexer评分 | `fp8_fp4_mqa_logits` | Q·K→ReLU→head加权求和，输出FP32候选logits。 | CSA prefill；Q/K格式由indexer_kv_dtype决定。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/sparse_attn_indexer.py#L484) |
| prefill top-k | `ops.top_k_per_row_prefill` / `topKPerRowPrefill` | 对有效候选区间选压缩记录ID，写int32 top-k buffer。 | prefill评分后。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/sparse_attn_indexer.py#L625) |
| decode分页评分 | `fp8_fp4_paged_mqa_logits` | 直接读paged量化K，结合context_lens与schedule metadata输出logits。 | CSA decode；spec decode可有next_n维和ragged pack/unpack。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/sparse_attn_indexer.py#L643) |
| decode top-k | `get_indexer_topk` / `cooperative_topk` / `persistent_topk` / `top_k_per_row_decode` | 按资格选择并写top-k IDs。 | auto依次尝试cooperative、persistent、per-row；另有显式DeepSelect/FI/Torch选择。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/indexer_topk.py#L236) |

评分主体为 score[t,j]=Σ_h weight[t,h]·ReLU(Q[t,h]·K[j])，随后按候选区间选K_i项。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/tests/model_executor/layers/test_mla_short_prefill_indexer.py#L1038) 该NVIDIA前向的Q/K融合kernel执行RoPE和量化，未显式调用Hadamard；不能把官方参考实现里的Hadamard直接列成本checkout的运行算子。FP8 Q的scale折进head权重，MXFP4 Q则另传packed scales。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py#L34)

#### [MoE] 路由、专家执行与共享专家

模型先按moe_backend选择原生MegaMoE、FlashInfer EP或普通FusedMoE。前两类共用模型专用mega routing，随后专家forward不同。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 路由元数据 | `prepare_mega_gate_routing_metadata` | 为hash层准备safe token IDs、hash/image mask；图像sentinel走视觉bias。 | mega家族模型每轮准备一次。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L107) |
| Router投影 | `GateLinear` / `bf16_mega_gate` | hidden[T,D]→FP32专家logits或直接top-k IDs/weights。 | mega家族：E=128时T≤1、其他E时T≤16用GateLinear；更大T用mega_gate。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1054) |
| sqrtsoftplus / hash top-k | `fused_topk_bias` / `dsv4_topk` / `ops.topk_hash_softplus_sqrt` | sqrt(softplus(logit))；bias辅助选ID，原score归一化并乘routed_scaling_factor。 | 小T mega路由和普通router可达；dsv4_topk仅其E/K/dtype等资格满足时。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py#L146) |
| EPLB映射 | `eplb_map_to_physical_and_record` | logical expert ID→physical replica ID，并记录负载。 | 原生路径按EPLB配置启用；FI EP拒绝EPLB。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L710) |
| 原生输入staging | `prepare_megamoe_inputs` | BF16 hidden→G32 FP8 + packed UE8M0，并写top-k IDs/weights及可选shared scale。 | 原生DeepGEMM MegaMoE；FI override不调用此Triton入口。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py#L163) |
| 原生融合专家 | `deep_gemm.fp8_fp4_mega_moe` | EP通信 + gate/up GEMM + clamp/SwiGLU×route + down GEMM + 写回/combine。 | deep_gemm_mega_moe；当前模型分支要求EP、FP4专家、SM10x及兼容shape。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L762) |
| FlashInfer EP专家 | `MoEEpMegaLayer.forward` / `kernel.stage_inputs` / `kernel.compute` | 将输入路由送入FI专用megakernel；DeepGEMM或NVFP4 CuTeDSL family。 | 两个flashinfer_moe_ep_mega_*配置；SM100/103及对应通信runtime。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/fi_moe.py#L313) |
| 普通FusedMoE专家 | `MoERunner` / `quant_method.apply` / `quant_method.apply_monolithic` | runner内生成gate logits，modular或monolithic kernel完成专家计算及通信。 | 非mega家族；expert_dtype fp4→MXFP4，显式NVFP4 override→ModelOpt；fp8→FP8方法。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L865) |
| 共享专家MLP | `gate_up_proj` / `SiluAndMulWithClamp` / `SiluAndMul` / `down_proj` | D→2SI→SwiGLU→SI→D，再与routed输出相加。 | 独立共享路径；原生MegaMoE满足完整资格时可融合shared，FI使用独立shared。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L185) |

路由专家的理想计算为 a=SiLU(xW_gateᵀ)⊙(xW_upᵀ)，贡献为 w·aW_downᵀ。原生MegaMoE把w放在L2前，并在加权后量化中间激活；最终combine不再乘w。详细流程见[源码导读](reference/mega-moe-source-guide.html)。

#### [输出] 主干收束、LM Head 与采样

主模型forward产hidden；compute_logits和Sampler属于之后的独立调用。runner通常只选择需要出token的hidden行投词表。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 多流head收缩 | `hc_head_fused_kernel_tilelang` | [T,C,D]→learned sigmoid权重→[T,D]，FP32关键统计。 | 最后PP rank；完成最后mHC post之后。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L1041) |
| 最终Norm | `RMSNorm` / `ir.ops.rms_norm` | 最后一维RMS归一化；保持[T,D]。 | hc_head之后；batch-invariant时选专用实现。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/layernorm.py#L74) |
| 词表投影 | `ParallelLMHead` / `LogitsProcessor._apply_head` | [T_selected,D]→词表logits；TP普通路径gather并去padding。 | generation默认model dtype；显式FP32 head_dtype时可torch.mm(out_dtype=FP32)。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/logits_processor.py#L136) |
| 采样与logprobs | `Sampler` / `TopKTopPSampler` / `argmax` | logits转FP32；按配置做penalty、temperature、top-k/top-p与随机或greedy采样。 | serving runner调用；speculation则进入验证/rejection路径。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/v1/sample/sampler.py#L73) |

compute_logits_local(skip_gather=True)返回本rank带padding的词表shard，不是完整全词表。采样前cast FP32也不意味着LM Head矩阵乘法默认FP32。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1999)

#### [通信] SP / TP / EP 的数据动作

通信可在模型层、runner或融合后端内部发生。按output_is_reduced等条件决定归约，不能把所有collective无条件叠加。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 本地token分片 | `sp_shard` / `sp_padding_mask` | padding + token维本地slice；本身不通信。 | PP=1、EP开启、TP>1，且mega家族或DP>1时启用本模型SP。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/common/ops/sequence_parallel.py#L42) |
| SP gather | `sp_all_gather` | Attention前收集token shard；模型末尾收集完整C路hidden。 | 本模型SP启用；优先custom collective，否则TP all_gather。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/common/ops/sequence_parallel.py#L23) |
| SP求和并分片 | `sp_reduce_scatter` | Attention TP partial结果求和并按token分片。 | SP启用；对应wo_b.reduce_results=False，避免再独立all_reduce。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/common/ops/sequence_parallel.py#L30) |
| 普通MoE通信/归约 | `dispatch_router_logits` / `combine` / `tensor_model_parallel_all_reduce` | 按EP runner或internal MoE kernel完成token交换及输出归约。 | 受DP/SP、supports_internal_mk、output_is_reduced等条件控制；原生/FI mega内部执行专用通信。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L774) |

PP边界交付已重建的[T,C,D] IntermediateTensors，send/recv由执行器安排，model.py没有直接launch这些网络操作。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1626)

#### [可选] MTP、DSpark 与图片输入

下面是按模型类/请求触发的路径。普通纯文本主模型forward不会依次运行ViT、MTP、DSpark和confidence head。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| classic MTP输入 | `_FUSED_MTP_INPUT_RMSNORM_KERNEL` / `e_proj` / `h_proj` | embedding与上一轮C路hidden各自Norm/投影后相加，进入draft V4 decoder。 | DeepSeekV4MTP；position=0只屏蔽embedding分支。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/mtp.py#L152) |
| classic MTP输出 | `mhc_post_tilelang` / `hc_head_fused_kernel_tilelang` / `_MTP_SHARED_HEAD_RMSNORM_KERNEL` | forward保留flatten C路hidden；compute_logits才collapse、Norm和head。 | classic MTP投机解码；spec_step_idx选择对应predictor层。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/mtp.py#L276) |
| DSpark target处理 | `combine_hidden_states` / `precompute_and_store_context_kv` | target aux streams mean后的特征→main_proj/Norm；预算各draft层context KV并写cache。 | DSpark独立draft；与classic MTP保存的全C路buffer不同。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/dspark.py#L184) |
| DSpark heads | `DSparkMarkovHead` / `compute_confidence` | 前token的低秩Markov bias修正base logits；可选linear+sigmoid估计confidence。 | DSpark query块主干并行；小Markov head与采样依序推进。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/dspark.py#L424) |
| 图片patch投影 | `DeepseekV4PatchEmbed.forward` | [N,3,p,p]→flatten→带bias Linear→[N,Dv]。 | vision_n_layers>0、允许图片且请求有image input。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/vision.py#L81) |
| 视觉Attention | `DeepseekV4VisionAttention` / `MMEncoderAttention` | QKV→2D RoPE→图内双向dense Attention→O投影。 | 视觉塔启用；不是语言CSA/HCA或MoE。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/vision.py#L97) |
| 视觉MLP与Norm | `DeepseekV4VisionMLP` / `SiluAndMul` | RMSNorm + dense gated MLP + 普通residual。 | ViT block；视觉Linear显式quant_config=None。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/vision.py#L154) |
| 视觉对齐与合并 | `DeepseekV4Aligner` / `F.unfold` / `GELU` / `embed_input_ids` | spatial merge→Linear/GELU/Linear→语言D维；按perm放入prompt的图块位置。 | 当前V4 wrapper逐图调用ViT/Aligner；不把V4.1 packed graph入口当成本路径。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/vision.py#L341) |



### Prefill 与 Decode 的具体变化

| 模块 | Prefill | Decode |
|---|---|---|
| 主输入投影 | 大T GEMM；满足阈值可多stream并行投影 | 小T可能选择低延迟GEMM |
| Compressor | 先保存本次各token state；仅完整C4/C128边界产生摘要 | 一般追加少量state，跨边界时才写新摘要 |
| Indexer评分 | gather量化K到连续chunk，fp8_fp4_mqa_logits | fp8_fp4_paged_mqa_logits直接读分页K；可含next_n |
| Indexer top-k | top_k_per_row_prefill | cooperative / persistent / per-row等selector |
| FlashMLA聚合 | 反量化gather主KV→合并索引→flash_mla_sparse_fwd | flash_mla_with_kvcache直接读paged cache |
| FlashInfer聚合 | 调同一DSV4 API的prefill分支；SM120按至多4请求分chunk | 调同一DSV4 API的decode分支 |
| mHC | 大T通常拆post和prenorm投影 | 本rank T≤32且D%128=0可融合前段 |
| MoE | backend不因名字prefill自动改变；staging/GEMM依T选择配置 | 专用mega routing小T阈值可能选择GateLinear+top-k |

短上下文CSA可直接全选压缩历史，但CUDA capture时保留固定评分路径。prefill的token总量取决本轮调度/chunk，不受示例buffer容量1024永久限制。

### 后端和精度怎样决定算子

| 配置/硬件 | 实际路径与关键条件 |
|---|---|
| CUDA attention未显式指定 | major12默认FlashInfer SM120，其他CUDA默认FlashMLA |
| FLASHINFER_MLA_SPARSE_DSV4 | major12→SM120类，其余→SM100类；仍须满足依赖/硬件资格 |
| FLASHMLA_SPARSE / FLASHMLA_SPARSE_DSV4 | FlashMLA类，packed V4 cache；prefill和decode入口不同 |
| FlashInfer SM100 | plain BF16或per-tensor FP8 cache；mixed storage-offset indices |
| FlashInfer SM120 | packed V4 cache；SWA与extra两套indices；不沿用SM100 RopeQuant融合 |
| deep_gemm_mega_moe | 专用EP融合路径；当前模型分支要求normalized sqrtsoftplus routing与FP4专家，SM10x及兼容shape |
| flashinfer_moe_ep_mega_deep_gemm | FI EP的deep_gemm_mega family；不是直接复用原生Triton staging |
| flashinfer_moe_ep_mega_cutedsl | FI EP的nvfp4_cutedsl family；对应runtime与权重pack要求 |
| 普通FusedMoE | quant_config按expert_dtype选择MXFP4或FP8；显式moe_quant_algo=NVFP4有独立覆盖；selector再选具体kernel |
| Indexer默认 | FP8；indexer_kv_dtype=mxfp4才走MXFP4，NVIDIA要求SM100 family |

选择入口：[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1146)、[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L965)、[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/quant_config.py#L173)。

**三个融合条件需要单独看。** mHC post/pre的融合前段要求本rank T≤32且D%128=0，后面仍有big_fuse；FlashInfer SM100把逆RoPE/输出量化融进attention要求128个实际local heads、per-tensor FP8 cache与FP8 wo_a；原生MegaMoE共享专家融合还要求非FI、SP或TP=1、开关允许以及兼容API/权重/scale。资格失败时保留对应的拆分路径。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L814) [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L115) [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L955)

**精度与存储对应不同对象。** 通用block-FP8 Linear常用G128激活；原生MegaMoE staging是G32 FP8，路由专家通常W4A8；主packed KV是448维G64 FP8+64维BF16尾部；Indexer FP8 cache每128维向量一个FP32 scale，MXFP4则每32维UE8M0 scale。mHC/Router的FP32统计或输出，也不代表全部参数与矩阵乘法都采用FP32。

### 其他平台与模型边界

| 平台入口 | 当前V4主要实现 | 适用边界 |
|---|---|---|
| NVIDIA | TileLang mHC、CUDA/Triton/CuTe融合、FlashMLA/FI、所选MoE | 上述详细清单的主体 |
| ROCm | AMD模型、AITER sparse MLA、AITER/其他MoE与mHC分支 | 独立硬件实现；不能套NVIDIA DeepGEMM/NVLink执行细节 |
| XPU | XPU Q/KV insert、FP8 decode dequant+sparse attention、xpu_fp8_bmm、_xpu_C mHC | VL入口当前为NotImplemented stub |
| CPU | CPU fused cache/compressor/indexer/MLA、BF16 packed O BMM、CPU mHC | Indexer为FP8-only；VL入口当前为NotImplemented stub |

平台入口按is_rocm→is_xpu→is_cpu→默认NVIDIA分支导出。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/__init__.py#L17)

本篇针对**V4**。V4也有DSpark；classic MTP、DSpark是两种独立draft路径。V4.1自己的Full/Reindex/Reuse、Engram、candidate selector、hc_collapse及视觉packed graph接入不应并入普通V4清单。当前V4视觉wrapper逐图eager调用共享ViT/Aligner；共享文件中存在的packed/DP helper不等于该wrapper已调用它。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/vl_model.py#L200)

### 源码定位与执行范围

源码快照：`vllm-project/vllm @ c91dccc0`。本篇按调用点追踪到vLLM自有CUDA、Triton、CuTe DSL、TileLang以及外部库native API，未把warmup注册、fake/meta实现、权重加载转换当成每轮算子。

| 阅读顺序 | 文件/入口 | 看什么 |
|---|---|---|
| 1 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1320) | decoder的mHC→Attention→mHC→MoE与延迟post |
| 2 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L471) | 投影、Q/KV准备、Indexer、聚合、O |
| 3 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/compressor.py#L354) | state、压缩边界、CuTe/Triton分支 |
| 4 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/sparse_attn_indexer.py#L321) | prefill/decode评分及top-k调度 |
| 5 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1038) | mega路由与普通FusedMoE的分界 |
| 6 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L814) | post/pre真正融合资格与多个kernel |
| 7 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/logits_processor.py#L136) | head dtype与词表投影 |

外部扩展内部的具体CUDA specialization不由这些Python入口唯一决定。本checkout的CMake固定DeepGEMM为vllm-project/DeepGEMM的e1f418c2、FlashMLA为0eee43b1；生成的FlashMLA接口和FlashInfer native实现不完整存放于当前树。本文确认到可读调用边界，不据此给出GPU实测性能或一次运行的完整launch trace。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/cmake/external_projects/deepgemm.cmake#L31) [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/cmake/external_projects/flashmla.cmake#L19)

<!-- vllm-runtime-end -->

## 本轮合并的关键更正

1. **特征投影与历史压缩分开。** 4096→512 改的是每 token 宽度；r=4/128 改的是长期槽密度。
2. **CSA 压缩有重叠。** 每 4 token 产出一槽，稳定阶段覆盖前后共 8 token；首槽只有 4 个有效 token。
3. **两套压缩器的顺序分开。** V4 的索引 K 先生成后评分；主 KV 在参考调用中可随后生成，但必须在正式 attention 读取前写好。
4. **先屏蔽未来槽，再 top-k。** prefill 计算图中存在原始分数，并不表示该 query 可访问未来槽。
5. **Indexer 合并 head 后选一次。** 各主注意力头读取共同地址，分别重算 attention 权重。
6. **专家编号与权重分开。** 查表路由按 token ID 选编号；分数路由按 `s+bias` 选编号；混合系数从不含 bias 的 `s` 中取得。
7. **缓存数值格式与物理分配分开。** BF16 量化模拟不能直接按 packed FP4 字节统计；V4.1 的 890 B/token 也不包含整套运行状态。
8. **模型规则与 GPU 后端分开。** MegaMoE 执行已有的路由任务，时间轴展示依赖，不把示意时长当成实测速度。

## 以后怎样增加内容

每次新增内容先归入现有章节，并给出适用模型、固定版本、输入输出形状、证据来源和一个可核对例子。新模型只有机制确实不同时才新增差异小节；共同机制继续复用现有章。一个问题只有一处主解释，其他位置用交叉引用。

HTML 是带动画的阅读版，Markdown 是可检索的正文版，二者由同一套章节源文档生成。静态文档不嵌入视频。动画使用缩小后的教学数据，支持暂停、逐步和重置；源码链接需要联网，正文与动画可离线使用。

## 历史资料如何处理

旧的独立 Flash/Pro/V4.1 全文、技术分析、核验更正与临时笔记保留作备查。发生冲突时，应回到固定版本源码核对，以本手册合并后的口径阅读。旧 Markdown 中失效的本机链接不会继续复制到新手册。

仓库中的个人 `notes/` 属于独立笔记，保持独立入口。今后 DS 系列的学习主线从新手册开始，不再要求读者把多份“完整版”拼起来。

