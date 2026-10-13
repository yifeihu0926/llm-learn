# DeepSeek V4.1 Flash · 完整算子解析

架构、计算公式、状态更新与推理实现

版本：2026-10-10。按本地固定配置与参考源码解释本模型；从整体架构逐步展开到各模块、公共内核及生成控制。

## 1. 整体架构与层类别

### 1.1 先看输入到输出

输入可以包含文本与图像；图像先经过视觉塔。主干有 40 个 Block，前 20 层为 CED Encoder、后 20 层为 Decoder；两段仍组成因果语言主干。Engram 在层 1、14 加入条件记忆，DSpark 从层 37、38、39 捕获特征并进行三阶段草稿预测。

**图 1：完整模型的主路径与辅助分支**

![完整模型的主路径与辅助分支]([本地源码路径])

最终主干状态先从 `[b,s,4,5120]` 收缩为 `[b,s,5120]`，词表输出宽度为 129,280。主干一次前向和辅助预测有独立调用入口，参数规模与每 token 激活量也应分别理解。

### 1.2 为什么一个 token 有四路残差

Embedding 只产生一个 5120 维向量，随后复制成四份，形状为 `[b,s,4,5120]`。它们属于同一个序列位置的内部状态，初始相同，之后通过不同的混合和回写权重逐渐形成不同表示，没有预先指定的四种语义分工。

每个 Attention/MoE 子层先加权收缩到单路 5120 维，再计算一次子层，最后把输出注入四路并混合旧残差。四路提供可学习的信息保留路径，主 Attention/MoE 的计算与参数并不直接乘四。`hc_mult=4` 是本配置的架构选择。

**图 2：一个标准 Block：Attention、MoE 与四路残差**

![一个标准 Block：Attention、MoE 与四路残差]([本地源码路径])

普通 Block 的执行顺序是：收缩 → Attention RMSNorm → Attention → mHC 回写 → 收缩 → FFN RMSNorm → MoE → mHC 回写。输入输出都是四路状态。 V4.1 的 pre 系数沿子层传递，下一子层使用本次产生的系数；当前 post/comb 用于当前回写。

### 1.3 特殊 Attention Block 的类别

**图 3：注意力分支与层职责**

![注意力分支与层职责]([本地源码路径])

Full 生成共享全局 KV、索引 K 和位置列表；Reindex 用本层 query 重新评价共享 K；Reuse 读取最近发布的列表。每层仍执行自己的 Q、局部 KV、主注意力、O 投影和 MoE。ratio 2/1 是全局粒度，Full/Reindex/Reuse 是状态职责；CED 也不是传统双向文本 encoder 加 cross-attention decoder。

| 类别 | 主干层号 | 数量 |
|---|---|---:|
| 纯 SWA | 0、1 | 2 |
| Full | 2、8、14、20 | 4 |
| Reindex | 24、28、32、36 | 4 |
| Reuse | 其余 | 30 |

2–19 使用 ratio 2，20–39 使用 ratio 1；辅助 DSpark 阶段占配置中的层号 40–42，ratio 0。`compress_ratios` 有 43 项，但主干层数是 40。

### 1.4 本模型的关键配置

| 配置 | 数值 |
|---|---:|
| 主干宽度 / 层数 | 5120 / 40 |
| 主注意力头数 / 每头宽度 | 64 / 512 |
| 非 RoPE / RoPE 宽度 | 448 / 64 |
| Q 低秩宽度 | 1280 |
| O 分组 / 每组低秩宽度 | 8 / 1024 |
| 专家总数 / 每 token 路由专家 / 共享专家 | 384 / 6 / 1 |
| 专家中间宽度 / 路由缩放 | 2304 / 1.5 |
| Indexer 头数 / 每头宽度 / top-k | 32 / 128 / 512 |
| 窗口 / mHC 路数 / Sinkhorn 轮数 | 128 / 4 / 20 |
| 语言 RMSNorm ε | 1e-20 |
| 词表大小 | 129280 |

上下文扩展基准为 65,536，YaRN factor=16，对应 1,048,576 的模型配置。运行时 `max_seq_len` 是单独的缓存容量，参考默认值 4096，需要按使用长度配置。后面的预算按数组或指定生产打包格式计算。

### 1.5 算子阅读路径

先沿第 2 章追踪语言 Block 内的输入、mHC、Q/KV、压缩/共享、Indexer/selector、注意力聚合、O、残差回写、路由和 SwiGLU；再读辅助分支与公共内核。第 7 章展开 RMSNorm、位置变换、量化/GEMM、online softmax、通信和缓存状态，最终覆盖 logits、采样与 EOS 控制。

## 2. 语言主干完整算子

本章按 V4.1 Flash 的实际前向顺序，从 token 表示一直讲到词表 logits。记 `b` 为 batch、`s` 为当前输入长度、`t` 为从 0 开始的绝对位置；主干宽度 `d=5120`，残差路数为 4，主注意力有 64 个 512 维头，索引注意力有 32 个 128 维头。所有层号均从 0 开始。

### 2.1 输入与 mHC：一份表示，四路残差

文本 token ID 经 `[129280,5120]` 的 Embedding 查表，得到 `[b,s,5120]`。有图像时，先把对应位置替换成 ViT/aligner 输出及图像边界标记的表示，再扩展成四个副本：

```text
token ID [b,s]
  → Embedding / 图像位置替换 [b,s,5120]
  → repeat [b,s,4,5120]
```

四路是同一个序列位置的四份内部状态，既没有增加 token 数，也没有让 Attention/MoE 执行四遍。它们在不同的混合与回写权重作用下逐渐形成不同表示。每个昂贵子层仍接收单路的 5120 维输入。层 1、14 的 Engram 在本层 Attention 之前向四路残差加入条件记忆；图像位置被排除在这项 n-gram 记忆贡献之外。

对单个 token 的四路状态 \(X\in\mathbb R^{4\times5120}\)，每个 Attention/FFN 子层分别有一套系数投影。将四路展开为 20480 维，使用一个覆盖整个 20480 维的 RMS 统计量，投影出 24 个数：

$$
z=\frac{W_{\rm hc}\operatorname{vec}(X)}{\sqrt{\operatorname{mean}(\operatorname{vec}(X)^2)+10^{-20}}},\qquad W_{\rm hc}\in\mathbb R^{24\times20480}.
$$

将 (z) 分成 4、4、16 维，生成收缩系数 (p)、回写系数 (q) 与混合矩阵 (B)：

$$
p=\sigma(s_pz_p+b_p)+10^{-6},\qquad q=2\sigma(s_qz_q+b_q),\qquad B=\operatorname{Sinkhorn}_{20}(\operatorname{reshape}(s_Bz_B+b_B)).
$$

只有 (B) 做 Sinkhorn。实现先做稳定的行 softmax 加稳定项、列归一化，再重复行/列归一化；得到近似双随机的 \(4\times4\) 混合矩阵。(p,q) 都不要求总和为 1。

**V4.1 的关键是：本子层生成的 (p) 供下一个子层使用。** 初始 (p=(1,0,0,0))，选择第一份副本。Attention 使用上一层 FFN 留下的 (p)，本次 Attention 产生的 (p) 供本层 MoE 收缩；本次 MoE 产生的 (p) 供下一层 Attention。当前生成的 (q,B) 则立即用于当前子层回写。

![mHC：收缩计算路径，保留四路状态]([本地源码路径])

沿图中的 pre 线先把四路合为单路；F 表示一次 Attention 或 MoE；post 将新结果注入四路，comb 混合原有四路状态。读 V4.1 时，要把 pre 理解成上一子层已经传来的系数。

### 2.2 主 Q 与局部 KV：每层自己的注意力入口

四路收缩并经 Attention RMSNorm 后，记输入为 \(x\in\mathbb R^{b\times s\times5120}\)。主 Q 使用两级低秩投影：

```text
qr = RMSNorm(wq_a(x))        [b,s,1280]
Q  = reshape(wq_b(qr))      [b,s,64,512]
Q 最后 64 维施加 RoPE
```

`wq_a` 的逻辑矩阵为 `[1280,5120]`，`wq_b` 为 `[32768,1280]`。V4.1 保留低秩空间的 `q_norm`，**展开成各个头后没有额外的逐头 Q RMSNorm**；也没有 Hadamard 变换。每个头的前 448 维不旋转，后 64 维承载 RoPE 位置变换。

局部 KV 是各层自己的 512 维共享 latent，而不是为 64 个头分别存一份 KV：

```text
local_KV = RMSNorm(wkv(x))  [b,s,512]
→ 最后 64 维 RoPE
→ 整个 512 维向量做 FP8 模拟量化
→ 写入局部缓存
```

V4.1 的局部量化包含 RoPE 尾部。prefill 使用当前输入的局部 KV，并为 decode 初始化长度为 128 的环形缓存；decode 每步把位置 (t) 写到 `t % 128`，读取最近至多 128 个有效位置。初期尚未填满的槽与未来位置由下标规则排除。纯 SWA 层 0、1 只读取这条局部路径；其他层把它与选中的全局路径合并。

RoPE 在纯 SWA 层使用基础 `theta=10000`；有全局 latent 的层使用压缩路径的 `theta=160000` 与 YaRN 配置，原始长度 65536、扩展因子 16。位置变换改变参与点积的表示，不能替代因果 mask。

### 2.3 Full、Reindex、Reuse：共享全局状态，保留本层计算

V4.1 Flash 的 40 个 Block 中，只有 4 层产生全局 KV，另外 4 层刷新全局位置选择。其余层复用这些状态，但每层仍执行本层的 Q 投影、局部 KV、主注意力和 O 投影。

| 注意力职责 | 层号 | 本层产生的全局状态 |
|---|---|---|
| 纯 SWA | 0、1 | 无全局 KV/K 或全局索引 |
| Full | 2、8、14、20 | 全局 KV、128 维索引 K、选择列表；20 还发布候选 mask |
| Reindex | 24、28、32、36 | 使用共享索引 K，本层 query/权重重新生成选择列表 |
| Reuse | 其余 30 层 | 使用已发布的全局 KV/K 与最近的选择列表 |

![40 层的全局 KV 来源与索引刷新位置]([本地源码路径])

层 2 的全局状态被 2–7 使用，层 8 的被 8–13 使用，层 14 的被 14–19 使用。20–39 共享层 20 的全局 KV/K；选择列表分别在 20、24、28、32、36 刷新，随后沿层传递。Reuse 复用的是全局位置列表，各层仍以自己的主 Q 计算实际注意力权重。

前半段 0–19 称 Encoder，后半段 20–39 称 Decoder，这是同一个因果语言主干内的 CED 划分。它不是独立双向 encoder 与 cross-attention decoder 的组合。Encoder 中 2–19 的全局粒度为 ratio 2；Decoder 为 ratio 1。

![CED：全长编码与尾部解码]([本地源码路径])

图示表达生产 bounded replay 的使用方式：共享全局状态支撑历史访问，局部尾部可被有限重算。下文算子按本地参考完整 forward 说明；参考实现仍遍历全长，不能把生产尾部 replay 的成本直接套到它。

<!-- attention-motion:v41 -->

### 2.4 全局 latent：ratio 2 聚合与 ratio 1 投影

**ratio 2** 的 Full 源层 2、8、14 把非重叠的相邻两个 token 合成一个 512 维 latent。每个 token 分别生成内容向量与 gate 向量：

$$
v_i=W_vx_i,\qquad g_i=W_gx_i,\qquad \pi_{j,i,a}=\frac{\exp(g_{i,a})}{\sum_{u\in\{2j,2j+1\}}\exp(g_{u,a})},\qquad c_j=\operatorname{RMSNorm}\left(\sum_{i\in\{2j,2j+1\}}\pi_{j,i}\odot v_i\right).
$$

这里 (a) 表示特征维度：softmax 沿两 token 的位置轴计算，每个特征维可以有不同权重。内容与 gate 投影、池化在 FP32 中计算，池化结果转回输入 dtype 后再 RMSNorm。没有重叠块，也没有额外的 learned positional term。

decode 的第一个 token 先留在未完成块 state，第二个 token 到达后才发布该摘要。prefill 尾部若剩一个 token，也保存在 state 等待后续完成。因此查询 (t) 可见的全局位置数为 \(\lfloor(t+1)/2\rfloor\)，而不是对所有历史位置向上取整。

**ratio 1** 的层 20 每个 token 直接生成一个全局 latent：

$$
c_t=\operatorname{RMSNorm}(W_vx_t).
$$

该路径使用 BF16 投影，没有块内 gate/pooling，也没有等待成组的 state。ratio 1 表示“一 token 一 latent”，并非停用全局缓存或读取全部历史；读取哪些全局位置仍由 Indexer 决定。

Full 层先以**尚未 RoPE 和量化的** (c_j) 产生索引 K，再对主全局 latent 做 RoPE 和 FP4 模拟量化。摘要 (j) 的代表位置是 (jr)。主全局 KV 按 16 维分组，使用 E4M3 scale；索引 Q/K 按 32 维分组，使用 E8M0 scale。Reindex 层复用已经生成的索引 K，没有自己的 `wk` 或 K cache。

### 2.5 Indexer 与 selector：先排名，再选主注意力的读取位置

Indexer 是 Attention 内部的选择分支。它复用主 Q 的低秩状态 (q_r)，投影为 32 个 128 维索引 query；Full 源层把 512 维全局 latent 投影为所有索引头共享的 128 维 key：

```text
Q_index = FP4模拟(RoPE(wq_b_index(qr)))         [b,s,32,128]
K_index = FP4模拟(RoPE(RMSNorm(wk(c_preRoPE)))) [b,N,128]
```

`wk` 的逻辑矩阵为 `[128,512]`。该索引路径没有 Hadamard。query 以当前位置旋转，key 以 latent 的代表位置旋转。更深 Reindex 用自己的 query 和头权重评价同一份共享 K。

每个 query (t) 对全局位置 (j) 的分数为：

$$
\alpha_{t,h}=\frac{(W_\alpha x_t)_h}{\sqrt{128}\sqrt{32}},\qquad s_{t,j}=\sum_{h=0}^{31}\alpha_{t,h}\operatorname{ReLU}(Q^I_{t,h}\cdot K^I_j).
$$

**ReLU 只作用于点积。** 头权重 (alpha) 不做 softmax/sigmoid，不保证非负，索引分数也不是主注意力概率。张量并行时，各 rank 的索引头贡献经 all-reduce 合成总分数。

先把未完成或未来全局位置的分数置为 \(-\infty\)，再选最多 512 个位置。V4.1 的实现选中 top-k 后**按位置编号排序**，用于读取缓存；它并未按分数降序输出。有效摘要下标还要加上局部 KV 占用的 offset：参考 prefill 中为当前局部长度 `s`，decode 中为 128。无效未来选择写成 `-1`。这些 ID 只控制 gather，主 value 的实际权重随后由 512 维主注意力计算。

**Decoder 的候选池是额外的一次块级筛选。** 层 20 把 ratio 1 的全局位置每 8 个分成一块，块分数取块内最高位置分数：

$$
u_{t,m}=\max_{j\in[8m,8m+7]}s_{t,j},\qquad \mathcal C_t=\bigcup_{m\in\operatorname{TopK}_{2048}(u_t)}\{8m,\ldots,8m+7\}.
$$

源码将包含最新有效位置的块临时置为最高优先级，使它占据所选至多 2048 块中的一个名额；未到达块被排除。展开到位置后，候选上限为 \(2048\times8=16384\)。最新部分块被保留，并不允许读取其中未来位置。

![分层候选池：先选块，再选择位置]([本地源码路径])

层 24、28、32、36 在同一候选 mask 内，用各自 query/头权重重新评分并取至多 512 个全局位置。层 20 自己的 top-k 来自它的因果有效位置分数；后续 Reindex 才把候选 mask 加到自己的分数上。相邻 Reuse 层直接使用最近发布的 top-k 列表。

本地参考代码仍以 dense einsum 计算全历史索引分数，然后施加 candidate mask；候选范围受限不等于该参考代码已经把打分量限制为常数。生产 backend 若采用候选 gather 后评分，才会减少更深层实际扫描的范围。

### 2.6 稀疏注意力、sink 与分组输出投影

局部有效 ID 与选中的全局 ID 拼成一个列表，交给一次 `sparse_attn`。对每个主注意力头，512 维 latent 同时作为 K 和 V。局部与全局记录在**同一个分母**内归一化：

$$
a_{t,h,j}=\frac{Q_{t,h}\cdot KV_j}{\sqrt{512}},\qquad O_{t,h}=\frac{\sum_{j\in\mathcal A_t}\exp(a_{t,h,j})KV_j}{\exp(\mathrm{sink}_h)+\sum_{j\in\mathcal A_t}\exp(a_{t,h,j})}.
$$

这里 \(\mathcal A_t\) 是当前可读局部与全局记录的列表。可学习的 `sink_h` 只增加分母，不提供非零 value，等价于一个 zero-value 槽。无效 ID 为 `-1` 时，value 为 0、logit 为 \(-\infty\)。内核沿索引列表分批 gather，使用运行最大值、运行分母和运行加权分子计算在线 softmax，避免保存完整的注意力矩阵。

得到 `[b,s,64,512]` 的 O 后，先对它的 RoPE 尾部执行逆旋转，再进入分组低秩输出投影。8 个组各自处理 8 个头，因此每组输入宽度为 4096：

```text
O [b,s,64,512]
  → 逆 RoPE
  → reshape [b,s,8,4096]
  → 每组自己的 wo_a [1024,4096]
  → [b,s,8,1024]
  → concat [b,s,8192]
  → wo_b [5120,8192]
  → Attention 输出 [b,s,5120]
```

`wo_a` 只连接本组的头，是按组隔离的投影；`wo_b` 再合并各组结果并恢复主干宽度。至此才完成 Attention 子层，Indexer 早已在其内部完成选择。

### 2.7 Attention 后面：mHC 回写，再进入 MoE

Attention 的 5120 维输出不会立刻进入词表 Head。它先与 Attention 入口保存的旧四路状态混合：

$$
X^A_i=q^A_iA(x)+\sum_{j=0}^{3}B^A_{j,i}X_j.
$$

其中 \(A(x)\) 已包含主注意力聚合、逆 RoPE 和 O 投影。然后更新后的四路状态 \(X^A\) 经 Attention 本次生成的 \(p^A\) 收缩，做 FFN RMSNorm，进入 MoE：

$$
x^F=\operatorname{RMSNorm}_{\rm ffn}\left(\sum_{j=0}^{3}p^A_jX^A_j\right),\qquad y=\operatorname{MoE}(x^F).
$$

FFN 的 \(q^F,B^F,p^F\) 从 \(X^A\) 生成：前两者用于当前 MoE 回写，最后一个传给下一层。回写仍以 MoE 入口保存的 \(X^A\) 为残差：

$$
X^{\rm next}_i=q^F_iy+\sum_{j=0}^{3}B^F_{j,i}X^A_j.
$$

![Attention 输出后的完整路径]([本地源码路径])

图中的两次回写分别属于 Attention 和 MoE。普通 Block 的完整顺序为“Attention → 回写四路 → 收缩与 FFN RMSNorm → MoE → 再回写四路”。只有第 39 层之后才执行最终收缩和词表输出；每层的 RMSNorm 都在对应昂贵子层之前。

### 2.8 MoE：路由、SwiGLU、图像选择偏置与输出合并

V4.1 Flash 主干每层有 384 个路由专家，单个 token 选择 6 个，并始终执行 1 个共享专家。它**没有前三层 token ID hash 专家路由**：主干各层都用当前隐藏态评分。Engram 的 n-gram hash 是另外的记忆查表机制，与专家路由无关。

对 FFN 输入 \(x^F\)，router 在 FP32 中投影，经 `sqrtsoftplus` 得到正分数：

$$
z_e=\frac{(W_rx^F)_e}{T_g},\qquad r_e=\sqrt{\operatorname{softplus}(z_e)},\qquad \operatorname{softplus}(z)=\log(1+\exp(z)).
$$

本地参考配置未另设 `gate_temp`，因此 \(T_g=1\)。实现使用稳定 softplus 运算。选择偏置按输入类型切换：文本位置使用普通 bias，图像 span 内的视觉位置及标记使用 VL bias。选中的 6 个专家与归一化权重分别为：

$$
\mathcal E_t=\operatorname{TopK}_6(r+b_{\rm text/VL}),\qquad w_e=1.5\frac{r_e}{\sum_{u\in\mathcal E_t}r_u+10^{-20}}\quad(e\in\mathcal E_t).
$$

**bias 只影响选中谁，不进入最终混合权重。** 六个路由权重之和近似 1.5；共享专家独立相加，不参与这组 top-k 归一化。输入图像也不会改变主干每 token 选 6 个路由专家的数量。

每个专家中间宽度为 2304，包含 gate/up 两个输入投影和 down 输出投影：

$$
g=\min(W_1x^F,10),\qquad u=\operatorname{clip}(W_3x^F,-10,10),\qquad a=\operatorname{SiLU}(g)\odot u,\qquad E(x^F)=W_2a.
$$

\(\operatorname{SiLU}(g)=g\sigma(g)\)。gate **只截上界**，up 截上下界；不能把两者都写成对称 clip。矩阵逻辑形状分别为 `[2304,5120]`、`[2304,5120]`、`[5120,2304]`。输出恢复到 5120 维，最终合并为：

$$
\operatorname{MoE}(x^F)=E_{\rm shared}(x^F)+\sum_{e\in\mathcal E_t}w_eE_e(x^F).
$$

参考代码把路由权重乘在专家中间激活上，再做 down 投影；以上公式表达其线性加权关系。分布式参考路径分别计算本 rank 的专家，再 all-reduce 路由输出，并加上共享专家。生产框架可用 MegaMoE/FusedMoE 执行打包、通信、专家 GEMM 和合并；这些是 MoE 的执行后端，**不会在主干中增加一个额外模型层**，也不会改成每 token 执行全部 384 个专家。共享专家是否与路由专家融合，由后端与并行布局条件决定。

### 2.9 最终收缩、RMSNorm 与词表 Head

第 39 层 MoE 回写后的状态仍为 `[b,s,4,5120]`。主干使用**末层 FFN 已经返回的** \(p^{F,39}\) 做最终收缩：

$$
h_t=\sum_{j=0}^{3}p^{F,39}_{t,j}X^{40}_{t,j},\qquad \ell_t=W_{\rm vocab}\operatorname{RMSNorm}_{\rm final}(h_t),\qquad W_{\rm vocab}\in\mathbb R^{129280\times5120}.
$$

V4.1 没有额外独立的 `hc_head` 投影或 head 混合参数；最终收缩直接使用最后一个 FFN 的 pre。最终 RMSNorm 后，词表 Head 才把 5120 维表示变成 129280 个词表分数。

参考普通生成只取最后位置，返回 logits `[b,129280]`；全位置调用可以得到 `[b,s,129280]`。Head 的权重在 checkpoint 中为 BF16，参考参数以 FP32 保存并计算 FP32 logits；词表分片时先各自计算，再 all-gather 拼接词表维度。logits 经过生成流程的温度/采样得到下一 token，并进入下一轮模型输入。

主干另外在层 37、38、39 的 **Attention 入口之前**，对四路取均值并收集 3 份 5120 维特征，供 DSpark 草稿分支使用。这不是主词表输出，也不是每层预测 token；辅助草稿分支与普通主干输出具有各自的调用路径。

本章对应本地实现入口：[`Transformer.forward`]([本地源码路径])、[`Block`]([本地源码路径])、[`Compressor / Indexer`]([本地源码路径])、[`Attention`]([本地源码路径])、[`Gate / Expert / MoE`]([本地源码路径])、[`ParallelHead`]([本地源码路径])、[`sparse_attn / Sinkhorn`]([本地源码路径])。
## 3. CED 与长输入执行

### 3.1 CED 的前后半部分

CED 将层 0–19 组织为 encoder，层 20–39 为 decoder。Encoder 的三个 KV 源层使用 ratio 2，decoder 源层 20 使用 ratio 1。


层 20 的全局 latent 来自该层经过 `hc_pre` 和 `attn_norm` 的输入。层 19 输出仍是 `[b,s,4,d]` 残差，不能直接作为已经收缩的 `[b,s,d]` KV 源输入。

官方生产实现结合 bounded decoder replay，在长序列 prefill 中让后半部分仅处理尾部窗口；官方给出的约 8B prefill / 16B decode 活跃参数口径与这一执行方式相关。参考 `Transformer.forward` 则对完整输入循环执行全部 40 层，仅有跨层共享并不自动产生这项 prefill 省算。

### 3.2 Bounded replay 与前缀复用

Bounded replay 分为两种用途：

- Encoder replay：长期复用以全局 KV 为中心，重新计算命中边界附近的有限尾部，以恢复后续计算所需的局部状态。
- Decoder replay：利用 encoder 已产生的全局信息，只执行 decoder 尾部窗口，减少长提示词的后半部分 prefill 计算。

两者都是近似重建。多层局部注意力的间接感受野超过单层 128-token 窗口，只重算一个有限尾部不能保证与完整历史前向逐位相同；前缀命中边界也可能改变 suffix 的 encoder 状态。模型通过后训练适配这种运行方式，报告中的较小质量影响来自作者实验。[V4.1 技术报告 §3.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

当输入长度远大于窗口时，减少 decoder prefill token 数能显著降低计算；短输入、缓存命中方式、执行后端和额外状态维护都会影响实际收益。



### 3.3 共享状态与实际省算的边界

参考 forward 对全长执行 40 层。生产 bounded replay 可在前半段处理完整输入，再使后半段只处理有限尾部，同时复用全局历史。按 token×层数概括是 `20N+20min(N,128)`，但每层投影、压缩、索引和注意力成本不同，实际性能需后端测量。

尾部重算近似重建局部状态，不保证与完整 forward 逐位相同。prefix caching 可重用已匹配前缀的状态；共享源层、prefix reuse、bounded replay 分别改变状态所有者、重复请求计算与 decoder 实际工作范围。

## 4. Engram：从局部 token 模式到四路条件记忆

Engram 位于主干第 1、14 层的入口，先更新四路残差，再执行该层 Attention。Attention 从当前序列的 KV 中读取上下文，Engram 则根据以当前位置结尾的 2、3、4-gram，读取训练得到的大型记忆表。每个位置只读取有限的表行，随后由当前隐藏状态决定各路接收多少记忆。

### 4.1 Token 映射与 n-gram 边界

前向的第一步是 `token_map[input_ids]`，将原始 token id 映射到 99,092 个压缩 id。这个 map 在模型初始化时构建：逐 token 解码，依次做 Unicode NFKC、NFD、去重音、小写、空白合并和首尾空白处理；遇到局部 UTF-8 字节解码产生的替换字符，则保留原始 token 字符串作为键。因此，归一化后相同的 token 可以共用哈希输入。

压缩 id 写入历史缓存。对位置 t，模型取出当前 token 和最多三个之前的 token，组成以 t 结尾的 n-gram。序列起点之前的历史用 pad id 补齐；图像段记为 `DEAD=-1`，回溯一旦遇到 DEAD，就把更早的回溯位置也视为边界外。因此，一个文本 n-gram 不会跨越图像段。历史缓存让 prefill 与后续逐 token decode 使用同一套规则。

### 4.2 整数哈希与分片 FP8 查表

每层对 2、3、4-gram 分别使用 8 个哈希头，合计 24 个索引。令 c 为压缩 token id，a 为固定的层专属奇数乘数，p 为素数桶大小，o 为桶偏移：

$$
H_{\ell,n}(t)=\bigoplus_{i=0}^{n-1}\left(a_{\ell,i}c_{t-i}\right),\qquad j_{\ell,n,h}(t)=H_{\ell,n}(t)\bmod p_{\ell,n,h}+o_{\ell,n,h}.
$$

这里的 XOR 是按位异或。每个 n-gram 长度、哈希头都有独立的素数桶；偏移把这些桶放入同一个大表的不同区间。乘数由固定层种子生成，在初始化后保持不变，不是每个 token 重新随机选择。不同哈希头使用不同模数，降低某一桶碰撞对整体表示的影响。

两个 Engram 表分别有 384,006,168 和 384,016,682 行，每行 256 维。表按行分片：本卡范围外的索引暂时读第 0 行，再将该结果置零；属于各卡的真实行通过 all-reduce 合并。查表仅反量化读到的行，不会先把整个表展开为 BF16。

每行的 FP8 数据按连续 32 维共享一个 scale。对第 b 组、第 r 个元素：

$$
e_{j,32b+r}=\operatorname{BF16}\!\left(\operatorname{FP32}(w^{\rm FP8}_{j,32b+r})\,\operatorname{FP32}(s_{j,b})\right),\qquad 0\le r<32.
$$

24 行记忆拼接成 6,144 维向量，随后执行 `Linear 6144→25600`，拆为四路各 5,120 维 key，以及一个共享的 5,120 维 value。

**图 9：Engram 从 n-gram 查表到四路门控注入**

![Engram 的哈希、记忆投影与四路回写]([本地源码路径])

图左侧的三行对应 2、3、4-gram，各生成 8 个哈希索引；紫色表读取 24×256 维记忆。右侧投影产生四路 key 和共享 value。底部回写线表示：四路都可以收到同一份 value，但每路的 key 与门控不同，所以注入强度可以不同。

### 4.3 归一化点积、带符号平方根与残差回写

对第 i 路残差 Xᵢ，模型将两组可学习逐维参数相乘，得到 wᵢ，再计算归一化点积 aᵢ：

$$
a_i=\frac{\sum_{c=1}^{d}X_{i,c}w_{i,c}K_{i,c}}{\sqrt d\,\sqrt{\operatorname{mean}(X_i^2)+\epsilon}\,\sqrt{\operatorname{mean}(K_i^2)+\epsilon}},\qquad w_i=q_i^{\rm weight}\odot k_i^{\rm weight}.
$$

两个 RMS 统计分别沿该路自己的 d=5120 维计算。它们不把四路拼成一个 4d 维归一化，也不执行对 24 行的注意力 softmax。点积先做带符号平方根，再通过 sigmoid：

$$
g_i=\sigma\!\left(\operatorname{copysign}\!\left(\sqrt{\max(|a_i|,10^{-6})},a_i\right)\right),\qquad X'_i=X_i+g_iV.
$$

`copysign` 保留输入符号，包括带符号的零；正零对应的稳定项为 +0.001。图像 mask 会将该位置的四个 gate 都置零。整个路径保持四路残差数量不变；Engram 只是向已有状态加入条件记忆。

| 步骤 | 张量形状 |
|---|---|
| 输入 token IDs | `[b,s]` |
| 两个 Engram 层的哈希索引 | `[b,s,2,24]` |
| 当前层查表结果 | `[b,s,24,256]` |
| 拼接与 K/V 投影 | `[b,s,6144] → [b,s,25600]` |
| 四路 key、共享 value | `[b,s,4,5120]`、`[b,s,5120]` |
| 门控与回写 | `[b,s,4] → [b,s,4,5120]` |

两层合计每个文本 token 名义读取 48 行，即 12,672 B 的 FP8 数据和 scale；重复模式、缓存命中与批内复用会影响实际读取量。大表提供容量，有限行数的访问控制每次前向的工作量。

源码：[token 归一化与哈希状态]([本地源码路径])、[n-gram 整数运算]([本地源码路径])、[分片查表与反量化]([本地源码路径])、[记忆投影与门控]([本地源码路径])。

## 5. 视觉输入：像素、ViT、空间拼接与语言序列

视觉分支先把像素编码为二维空间特征，再减少视觉位置数，并映射到语言模型的 5,120 维。视觉塔内部使用单路残差；图像特征进入语言序列之后，才与文本一起扩展为四路 mHC 状态。

### 5.1 图像预算、像素归一化与 Patch projection

图像先转 RGB。预处理按长宽比、最小像素面积和每图 1,024 个语言位置的上限规划尺寸，将高度和宽度对齐 14 像素 patch；若预算超限，先收缩规划尺寸，再用 ImageOps.pad 按比例缩放并以灰色边缘补齐目标画布。

像素归一化使用 `((RGB/255)-0.5)/0.5`，范围变成 [−1,1]，转为 BF16。图像按行序切成 `[N,3,14,14]` patches；每块展平为 588 个值，经过带 bias 的 `Linear 588→1024`。

令 ViT 网格为 nₕ×n𝓌，经 3×3 Aligner 后，语言特征网格为 `h=ceil(nₕ/3)`、`w=ceil(n𝓌/3)`。图像序列还包含每行换行和首尾标记，因此预算是：

$$
N_{\rm image}=h(w+1)+2\le1024.
$$

预算中的 1,024 包含标记位置，不能只数 h×w 个视觉特征。

### 5.2 三十二层 ViT：二维 RoPE 与完整双向 Attention

ViT 的隐藏维度为 1,024，32 层，16 个注意力头，每头 64 维。每层先 RMSNorm，再联合投影 `1024→3072`，拆成 Q、K、V。Q/K 使用 patch 的行、列坐标生成二维 RoPE；全部 64 维被分配为行与列旋转对，V 不旋转。

对每一对通道，旋转仍是：

$$
\begin{bmatrix}q'_a\\q'_b\end{bmatrix}=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}\begin{bmatrix}q_a\\q_b\end{bmatrix}.
$$

视觉旋转相位分别来自行坐标和列坐标。随后对整幅图像执行完整双向 SDPA：

$$
P=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{64}}\right),\qquad O=W_o(PV)+b_o.
$$

同一图像的所有 patch 可以互相读取。这个分支不使用语言因果 mask、SWA、全局 KV 压缩器或 learned Indexer。视觉 Attention 的 QKV 与 O 投影都带 bias。

### 5.3 单路残差、SwiGLU 与最终视觉 Norm

视觉 Block 的两次残差加法为：

$$
x_1=x+\operatorname{VisionAttn}(\operatorname{RMSNorm}_1(x)),\qquad x'=x_1+\operatorname{VisionMLP}(\operatorname{RMSNorm}_2(x_1)).
$$

MLP 的中间维为 2,816：第一次无 bias 的 Linear 输出 5,632 维，拆成 gate 和 up，然后执行 SwiGLU，再无 bias 地映射回 1,024 维：

$$
\operatorname{VisionMLP}(z)=W_2\bigl[\operatorname{SiLU}(W_{1,g}z)\odot(W_{1,u}z)\bigr].
$$

视觉 MLP 没有 MoE 的专家路由和语言专家路径的 clamp。32 层之后还做一次 RMSNorm；所有视觉 RMSNorm 使用 ε=10⁻⁶，与语言主干的 ε=10⁻²⁰ 不同。

**图 10：视觉网格通过 Aligner 进入语言序列**

![ViT、3×3 空间拼接、Aligner 与图像 token 布局]([本地源码路径])

从图左侧向右看，ViT 保留细 patch 网格，3×3 空间拼接将九个相邻位置变成一个长向量，Aligner 将它映射成语言模型维度。底部绿色位置是视觉特征，蓝色位置是文本；图像特征、图像标记与文本沿同一个语言主干执行 prefill。

### 5.4 Aligner：Pad、Unfold、GELU 与维度映射

Aligner 将视觉序列恢复为 nₕ×n𝓌 网格，右侧和底部补零到 3 的整数倍，使用 `unfold(kernel=3,stride=3)` 将每个 3×3 区域拼接为 9×1024=9216 维。它不求平均，因此保留九个位置各自的通道值。

随后执行两次带 bias 的 Linear，中间使用 GELU：

$$
y=W_2\operatorname{GELU}(W_1z+b_1)+b_2,\qquad z\in\mathbb R^{9216},\quad y\in\mathbb R^{5120}.
$$

完整维度链是 `9216→5120→5120`。默认 GELU 使用 `0.5x(1+erf(x/√2))`，与 ViT MLP 的 SiLU 不同。

### 5.5 图像标记替换、Mask 与四路扩展

语言序列里的 IMAGE 槽按行序替换为 Aligner 输出；IMAGE_START、IMAGE_END、IMAGE_NEW_LINE 分别使用学习得到的 5,120 维标记向量。整段位置在 `input_ids` 中统一使用 `129264`，靠 `token_types` 区分标记和特征。

图像段必须完整包含在首次 `start_pos=0` 的 prefill 内。替换完成后，文本和图像 embedding 一起扩展为 `[b,s,4,5120]`。图像 mask 在 Engram 中切断 n-gram 并关闭记忆注入，在 MoE 中选择视觉路由 bias；它不会跳过语言 Attention 或 MoE。

源码：[图像尺寸与像素预处理]([本地源码路径])、[ViT 位置编码与 Block]([本地源码路径])、[Aligner]([本地源码路径])、[图像特征与标记合并]([本地源码路径])。

## 6. DSpark：草稿 Block、Markov 修正与输出控制

DSpark 是主干之外的三阶段草稿分支。它与主干共享 token embedding 和词表 Head，每个 stage 保留完整的 Attention、mHC 与 MoE。每阶段有 128 个路由专家、激活 3 个，并保留一个共享专家；主干则是 384 个路由专家、激活 6 个。

DSpark 的分工是：主干提供上下文特征，三个 stage 为五个草稿位置计算 hidden，最后用 Markov Head 将前一个确定 token 的信息加入各位置词表分数，并顺序采样。

### 6.1 捕获三层主干入口状态，融合为 main_x

主干在第 37、38、39 层执行之前捕获四路状态；先各自对四路取均值，再拼接。捕获点位于该层 `hc_pre` 和 `attn_norm` 之前，读取的是 Attention 入口处的四路状态，不是三个 Block 的输出。

$$
m=\operatorname{concat}\!\left[\operatorname{mean}_{\rm hc}(X_{37});\operatorname{mean}_{\rm hc}(X_{38});\operatorname{mean}_{\rm hc}(X_{39})\right],\qquad \operatorname{main\_x}=\operatorname{RMSNorm}(W_{\rm main}m).
$$

形状依次为 `[b,s,4,5120] → [b,s,5120] → [b,s,15360] → [b,s,5120]`。`main_proj` 位于第一个草稿 stage；得到的 main_x 提供给各 stage，但各 stage 用自己的 KV 投影权重编码它。

**图 11：三阶段草稿 hidden 与顺序 Markov 采样**

![DSpark 的主特征融合、草稿 Block 和顺序采样]([本地源码路径])

左侧三个捕获点融合成 main_x，顶部蓝色路径把上下文送给草稿 stage。右侧每行是一个完整 stage，每列是一个草稿位置：同一行的五个位置一起更新，三行沿深度依次执行。底部 anchor→t₁→…→t₅ 表示真正输出 token 时的 Markov 条件依赖。

### 6.2 Anchor、Noise 与 Prefill 缓存预热

草稿块大小为 5。输入序列为 `[anchor, noise, noise, noise, noise]`，noise token id 是 128799。共享 embedding 查表后，扩展成 `[b,5,4,5120]`，初始 `pre_mix` 为只选第 0 路的 one-hot。

首次 prefill 时，各 stage 仅从 main_x 生成 KV 并填充 128 槽窗口缓存，不执行完整草稿 Block、不计算草稿输出。后续 decode 才执行五位置草稿。输入的五个位置与输出的五个新 token 是两个计数：最终 IDs 包含 anchor，所以长度为 6。

### 6.3 DSpark Attention：历史窗口与非因果草稿块

每个 stage 对 main_x 执行 `Wkv→KV RMSNorm→RoPE→原位量化`，将当前主干上下文 KV 写入 `start_pos mod 128` 的环形槽。草稿 hidden 则执行自己的低秩 Q 投影、Q Norm、Q 展开、RoPE，以及 KV 投影、KV Norm、RoPE 和量化。

稀疏位置表对每个草稿 query 包含两部分：已填充的历史窗口槽，以及全部五个草稿位置。因而五位置草稿块内部是**非因果 Attention**，每个位置都能读取整个草稿块。这允许草稿位置共同更新 hidden；它没有 CSA/HCA 压缩历史，也没有 learned Indexer。

Sparse Attention 之后，输出执行 inverse RoPE，再按 8 组做低秩输出投影 `Wo_a`，拼接后通过 `Wo_b` 返回 5,120 维。随后执行 mHC 回写、下一子层的收缩和 Norm、草稿 MoE，以及第二次 mHC 回写。草稿 token 是文本，MoE 不使用 VL bias。

### 6.4 词表 Head、低秩 Markov Bias 与顺序采样

最后一个 stage 用传出的 `pre_mix` 收缩四路，RMSNorm 后调用共享词表 Head，得到 `[b,5,129280]` 基础 logits。Markov Head 另外维护 rank 256 的 embedding 与词表矩阵；对于当前已确定 token yᵢ：

$$
e_i^{M}=E_M[y_i]\in\mathbb R^{256},\qquad b_i=W_Me_i^{M}\in\mathbb R^{129280},\qquad \widetilde\ell_i=\ell_i+b_i.
$$

随后采样 yᵢ₊₁，再以这个新 token 生成下一个位置的 Markov bias。五个基础 hidden 可以整块计算，Markov bias 与采样仍有逐位置的顺序依赖。Markov 参数是单独的 rank-256 参数，不与主 embedding、主词表矩阵共享。

词表 Head 的 GEMM 在 FP32 中输出 logits；词表分片时先计算局部分数，再 all-gather，并沿词表维拼接。

### 6.5 Confidence Head 的输入与分数含义

每个位置还将收缩后的 hᵢ 与 Markov embedding 拼接，执行 FP32 的 `Linear 5376→1`：

$$
c_i=w_c^{\mathsf T}[h_i;e_i^{M}].
$$

hᵢ 是最后收缩后的 5,120 维状态，尚未经过最终词表 Head 之前的 RMSNorm。源码直接返回这个标量，没有 sigmoid；因此 cᵢ 是原始置信度分数，不能直接解释为 0–1 的接受概率。

| 返回张量 | 形状 | 内容 |
|---|---|---|
| output_ids | `[b,6]` | anchor 与 5 个新草稿 token |
| logits | `[b,5,129280]` | 已加入 Markov bias 的词表分数 |
| confidence | `[b,5]` | 未经 sigmoid 的原始分数 |

### 6.6 温度采样与生成循环

普通主干输出与草稿输出使用同类采样算子。温度 T=0 时取 argmax；正温度时先做 FP32 softmax，再执行 exponential-race：

$$
p_v=\operatorname{softmax}(\ell/\max(T,10^{-5}))_v,\qquad e_v\sim\operatorname{Exp}(1),\qquad y=\arg\max_v(p_v/e_v).
$$

这种采样与对应类别分布的多项式采样等价。参考路径没有额外的词表 top-p/top-k 筛选；Attention 选择历史位置与 MoE 选择专家的 top-k 都不是这里的输出采样。

普通生成循环还执行 prompt mask 的 `where` 覆盖、token 写入、EOS 比较、finished mask 的逻辑更新和 `all()` 停止。DSpark 的 `forward_spec` 返回草稿前向结果；完整推测解码还需要 serving 组织主模型校验、接受/拒绝、状态回滚及补采样。官方参考生成脚本只调用普通 `forward`，没有接入这套推测解码循环。

源码：[特征捕获与 forward_spec]([本地源码路径])、[DSpark Attention 与位置表]([本地源码路径])、[Markov 与 Confidence Heads]([本地源码路径])、[草稿初始化与输出]([本地源码路径])、[采样]([本地源码路径])、[生成控制]([本地源码路径])。
## 7. 公共算子与执行内核

本章沿 DeepSeek V4.1 Flash 的前向路径解释归一化、四路混合、位置处理、量化矩阵乘法、注意力聚合和状态更新。这里的“算子”指具有明确输入输出和计算作用的逻辑步骤；设备执行时可以把多个步骤融合在同一个 kernel 中。`reshape`、`gather` 等布局或索引步骤结合所属模块说明。

统一记号：b 为 batch，s 为当前输入长度，d 为主干宽度，H 为注意力头数，G 为输出投影组数，r_q/r_o 为 Q/O 的低秩宽度，N 为已提交的全局槽数。主语言注意力的 latent 宽度为 512，RoPE 尾部为 64；四路残差 shape 为 `[b,s,4,d]`。

### 7.1 RMSNorm、逐头标准化与张量布局

RMSNorm 对最后一维计算均方和倒平方根，再乘可学习的逐维权重 γ。输入和输出 shape 相同，不减均值：

```text
r = 1 / sqrt(mean(x², last_dim) + ε_norm)
y = γ × x × r
ε_norm = 1e−20
```

参考路径先转 FP32 做均方、rsqrt 和缩放，再转回输入 dtype。Attention/FFN 入口的 RMSNorm 处理收缩后的 `[b,s,d]`；低秩 Q 的 `q_norm` 处理 `[b,s,r_q]`；KV 的 RMSNorm 处理 `[b,s,512]`。归一化作用于哪个维度，要与当前 tensor 的 shape 一起读。

本模型只在低秩 Q 空间执行带权重的 q_norm，展开主头后不再逐头归一化；视觉塔另用 ε=1e−6。

`view/reshape/unflatten` 将投影结果解释成 head、group 或四路维度，`transpose` 调整维度次序，`flatten/concat/split` 合并或拆分特征。它们改变布局或表示方式；真正的特征混合由投影和加权计算完成。非连续存储在 `contiguous` 时可能发生复制。`cast` 则改变数值精度，应与纯布局变化区分。

源码入口：[模型中的 RMSNorm、Attention.forward 与 Block]([本地源码路径])。

### 7.2 mHC：系数投影、Sinkhorn、收缩与回写

每个 token 的四路状态 X 为 `[4,d]`。系数生成先展平为 `[4d]`，在整个 4d 上使用一个 RMS 统计量，投影为 24 维，拆成 4、4、16 三组：

```text
z = W_hc vec(X) / sqrt(mean(vec(X)²) + ε_norm)   [24]
(z_p, z_q, z_B) = split(z, 4, 4, 16)
p = sigmoid(s_p z_p + b_p) + ε_hc              [4]
q = 2 × sigmoid(s_q z_q + b_q)                 [4]
B = Sinkhorn20(reshape(s_B z_B + b_B, 4, 4))    [4,4]
ε_hc = 1e−6
```

p 用于收缩，q 用于注入子层结果，B 用于混合原有残差。只有 B 经过 Sinkhorn，p/q 各元素的和不要求等于 1。mHC 的 pre 系数在本模型中何时生成、由哪个子层使用，遵循前面标准 Block 的执行顺序；下面公式表示一次收缩和一次回写的计算本身：

```text
h = Σ_j p_j X_j                              [d]
u = Attention(RMSNorm(h)) 或 MoE(RMSNorm(h))  [d]
X'_i = q_i u + Σ_j B[j,i] X_j                [4,d]
```

`B[j,i]` 表示输入路 j 向输出路 i 的贡献，索引方向对应源码在输入路维度求和。批量计算时 p/q 为 `[b,s,4]`，B 为 `[b,s,4,4]`，h/u 为 `[b,s,d]`。

Sinkhorn 先做稳定行 softmax、加 ε，再做一次列归一化；随后重复 19 次“行归一化 → 列归一化”：

```text
B = row_softmax(logits) + ε_hc
B = B / (column_sum(B) + ε_hc)
repeat 19 times:
    B = B / (row_sum(B) + ε_hc)
    B = B / (column_sum(B) + ε_hc)
```

配置中的 20 轮和稳定项使 B 的行列和接近 1；有限精度下不保证严格等于 1。RMS 投影、sigmoid、行列归一化和加权和是四个不同步骤，内核将部分步骤融合并不会改变其逻辑关系。

源码入口：[模型中的 hc_mixes/hc_pre/hc_post 与 Block.forward]([本地源码路径])、[hc_split_sinkhorn 封装与内核]([本地源码路径])。

### 7.3 RoPE、YaRN、逆旋转与索引维度处理

RoPE 对主 Q/KV 最后 64 维里的每对实数做位置旋转，输入输出 shape 相同：

```text
φ_j = position × frequency_j
frequency_j = θ^(−2j / d_rope)
(a', b') = (a cosφ_j − b sinφ_j, a sinφ_j + b cosφ_j)
```

源码将每对实数视为复数并乘 `exp(iφ)`；逆旋转乘其共轭。纯 SWA 使用 θ=10000 并禁用 YaRN；正 ratio 的主干层使用 θ=160000 和扩展设置。YaRN 在频率维上用线性 ramp 混合原频率与缩放频率：

```text
frequency'_j = frequency_j × (1 − ramp_j)
             + (frequency_j / factor) × ramp_j
factor = 16，original_seq_len = 65536
```

ramp 的两端由 `beta_fast=32`、`beta_slow=1` 与原长度算出。压缩槽 j 的 RoPE 锚点为 `j×ratio`，即该组第一个 token 的位置；该组最后一个 token 决定摘要何时完成、何时允许读取。位置锚点与因果可见边界是两个概念。

注意力输出尾部 64 维按当前 query 位置做逆旋转，然后进入 O 投影。这一步位于聚合之后，不能在结构图中省略为直接输出原始 latent。

本模型的主注意力和 Indexer 都不做 Hadamard。索引 Q/K 先 RoPE 后执行 FP4 模拟量化。

源码入口：[precompute_freqs_cis、apply_rotary_emb 与 Attention 的位置处理]([本地源码路径])。

### 7.4 FP8/FP4、scale 与量化模拟

低比特张量的数值由量化值与 scale 共同表示。对一组 g，先计算幅值上界，再缩放、截断范围、转目标格式；反量化乘回 scale：

```text
a_g = max(abs(x_i), i∈g)
x_q[i] = cast_Q(clip(x_i / s_g, −Qmax, Qmax))
x_hat[i] = s_g × x_q[i]
FP8 E4M3: Qmax=448
FP4 E2M1: Qmax=6
E8M0 scale: s_g = 2^ceil(log2(a_g / Qmax))
```

FP8 amax 至少取 `1e−4`。未启用 scale rounding 时 scale 直接为 amax/448；启用时取 2 的整数次幂。FP4 E8M0 的 amax 下限为 `6×2^−126`。FP4 的两个量化值打包在一个字节内，scale 则是另一组 tensor，容量计算需要同时计入。

本模型量化 Linear 的激活沿最后一维每 32 个值一组；FP8 权重每 32×32 一个 scale tile。FP4 专家权重沿每个输出行的 K 方向每 32 个值一组，使用 E8M0 scale。量化 tile、历史压缩组和缓存页具有不同用途。

局部 KV 全部 512 维按 32 分组做 FP8，包括 RoPE 尾部；全局 KV 全部 512 维按 16 分组做 FP4，scale 为 `cast_E4M3(max(amax,6×2^−9)/6)`，不套用 E8M0 幂次公式。Indexer 的 128 维 Q/K 则每 32 值 FP4、使用 E8M0 scale。

`act_quant(..., inplace=True)` 或 `fp4_act_quant(..., inplace=True)` 在参考路径执行“量化 → 反量化 → 回写输入”，缓存 tensor 仍是 BF16。这种模拟对应低比特数值，不等于将该缓存的物理存储直接变成 FP8/FP4。实际低比特 Linear 权重和 GEMM 输入则有独立量化数组与 scale。

源码入口：[Linear 的权重格式与 Attention/Indexer 中的量化调用]([本地源码路径])、[act_quant、fp4_act_quant 及量化内核]([本地源码路径])。

### 7.5 Linear 与带 scale 的矩阵乘法

参考语言模型的 `linear()` 路径要求 bias=None，逻辑输入 `X[...,K]`，权重 `W[N,K]`，输出 `Y[...,N]`：

```text
Y = X Wᵀ
```

权重 dtype 决定执行普通 `F.linear`、FP8 激活×FP8 权重 GEMM，或 FP8 激活×FP4 权重 GEMM。FP4 专家权重不意味着其输入激活也为 FP4：这条 GEMM 路径先把输入激活量化为 FP8。

量化矩阵乘法需要在对应 K 分组累加时补双方 scale。令量化激活分组宽度 Q=32，两种 GEMM 的 scale 索引分别为：

```text
FP8 激活 × FP8 权重：g 是沿 K 每 Q 个值的分组
Y[m,n] = Σ_g Σ_(k=gQ…(g+1)Q−1)
         A_q[m,k] × W_q[n,k] × s_A[m,g] × s_W[floor(n/Q),g]

FP8 激活 × FP4 权重：g 是沿 K 每 32 个值的子块
Y[m,n] = Σ_g Σ_(k=32g…32(g+1)−1)
         A_q[m,k] × W_q[n,k] × s_A[m,floor(32g/Q)] × s_W[n,g]
```

FP8 权重的 scale 共享于 Q 个输出行和 Q 个 K 值组成的 tile。FP4 权重每输出行、每 32 个 K 值一个 scale：本模型 Q=32，两边的 K 分组一一对应。FP4 子块在参考内核中经过 FP32 转为 FP8 后参与 GEMM，再用双方 scale 补偿；不能漏掉其打包和 scale 索引。

参考量化 GEMM 采用 FP32 累加和 scale 补偿，再写出 BF16。虽然 Linear 的数学公式相同，低比特舍入、分块尺度和累加精度会影响有限精度结果。

源码入口：[linear、Linear、ColumnParallelLinear 与 RowParallelLinear]([本地源码路径])、[fp8_gemm、fp4_gemm 与内核]([本地源码路径])。

### 7.6 稀疏 gather、online softmax 与 attention sink

主注意力输入为 Q `[b,s,H,512]`、共享潜在 KV `[b,N_total,512]`、位置列表 `[b,s,K]` 和每个 head 一个 sink logit。此处 K 是待访问的位置数，N_total 包括当前可读的局部原始槽和全局槽；同一个 512 维 KV 向量同时用作 Key 和 Value。

窗口位置与全局位置合并后进入同一次 softmax 归一化：

```text
score[h,j] = dot(Q[h], KV[j]) / sqrt(512)
Z[h] = exp(sink[h]) + Σ_(j∈selected) exp(score[h,j])
O[h] = Σ_(j∈selected) exp(score[h,j]) × KV[j] / Z[h]
```

Sink 是一个 learned logit，只增加分母，等价附加 value 为零的一个注意力槽。它不增加新的历史 KV 内容。输入位置列表为 int32，`−1` 代表不可访问位置；内核为它读取零向量、赋 `−∞` score，使其不贡献分子或分母。因果边界由上游列表构造和 selector 保证。

内核每批 gather 64 个位置，使用 online softmax 维护运行最大值 m、分母 l 和向量分子 r，避免物化完整 `[s,H,N_total]` 注意力矩阵：

```text
m_new = max(m, max(score_this_chunk))
l_new = exp(m − m_new) × l + Σ exp(score − m_new)
r_new = exp(m − m_new) × r + Σ exp(score − m_new) × KV
最后：O = r / (l + exp(sink − m))
```

主乘法及运行累加包含 BF16 输入与 FP32 accumulator。online softmax 是上述注意力的分块执行方式，算完得到 `[b,s,H,512]`，随后逆 RoPE 和 O 投影。本地 head 不足 16 时封装临时补零到 16，再裁去辅助 head；padding 是执行布局选择，不改变模型头数。

源码入口：[Attention.forward 对 sparse_attn 的调用]([本地源码路径])、[sparse_attn 封装与 sparse_attn_kernel]([本地源码路径])。

### 7.7 分组低秩 O 投影与最终输出头

注意力聚合输出逆 RoPE 后，将 H 个 head 分为 G 组，每组先独立投影到 r_o，再拼接并投影回 d：

```text
O                         [b,s,H,512]
reshape_by_group(O)       [b,s,G,(H/G)×512]
u_g = W_Oa,g × concat(O_heads_in_group_g)  [r_o]
y = W_Ob × concat(u_0,…,u_(G−1))          [d]
```

`W_Oa` 为按组独立的投影，每一组只能读自己的 head，不是一张跨所有 head 的 dense 权重。参考转换流程将它变为 BF16，前向使用 `einsum`；`W_Ob` 使用 RowParallelLinear，分片结果合并后形成 `[b,s,d]`。主注意力的输出还需经过当前子层的 mHC 回写，才能成为下一个子层的四路输入。

最终收缩使用第 39 层 FFN 已返回的 pre 系数；没有独立 hc_head 投影。DSpark 最后 stage 也使用返回的 pre；其 confidence 接收最终 Norm 前的 collapsed hidden。

得到最终 `[b,s,d]` 后，RMSNorm 和词表线性头计算 logits：

```text
logits = W_vocab × RMSNorm(h_final)    [b,s,V]
普通逐 token 生成只读取最后位置         [b,V]
```

每个位置都有对应的逻辑词表投影；公开入口实际返回哪些位置，以本模型前面的输出头说明为准。普通生成使用最后位置的 logits，词表分片时各卡计算自己的区间，再 gather 合并。

源码入口：[Attention 的 grouped O、最终收缩、ParallelHead 与 Transformer.forward]([本地源码路径])。

### 7.8 并行通信：分片输出如何合并

计算分片决定是否需要通信，不能看到一个 Linear 就假定它都做 all-reduce。

| 位置 | 分片方式与 shape | 参考执行 |
|---|---|---|
| 词表 Embedding | 每卡持有 `[V/P,d]` | 非本卡词表区间的结果置零，all-reduce 求和得到 `[b,s,d]` |
| ColumnParallelLinear | 按输出维切分 | 本卡输出已经是该分片的完整结果，当前入口不做 all-reduce |
| RowParallelLinear | 按输入 K 维切分 | 本卡得到部分输出和，转 FP32 all-reduce，再转回输入 dtype |
| Indexer | head 分片 | 不同卡贡献的索引 score 相加后再选位置；源码没有统一先转 FP32 的步骤 |
| 路由专家 | 每卡持有一组专家 | 本卡专家输出在 FP32 累加，all-reduce 后加入共享专家输出 |
| 词表 Head | 每卡持有 `[V/P,d]` | 本卡 logits `[b,V/P]` 用 all-gather 后沿词表维拼接为 `[b,V]` |

以上是公开参考实现的通信路径。其 MoE 采用本地专家计算、结果 all-reduce 的写法；源码没有 `all_to_all`。生产 MegaMoE 的 token dispatch/combine、通信融合和调度属于专家执行章节，不能套成这份参考 forward 的实际调用。

源码入口：[ParallelEmbedding、并行 Linear、Indexer、MoE 与 ParallelHead]([本地源码路径])。

### 7.9 缓存布局、状态更新与跨步一致性

缓存算子既包括写入/读取，也包括逻辑位置与物理槽之间的映射。它们不引入新的学习参数，但直接影响因果边界和 prefill/decode 一致性。

| 状态 | shape 或索引 | 更新与读取 |
|---|---|---|
| 窗口 KV | `[b_max,128,512]` | decode 将 token t 写入 `t%128`；历史逻辑位置通过同一环形映射读取 |
| prefill 的窗口输入 | 当前输入 `[b,s,512]` | 本次计算直接读当前 chunk；另外保存最后 128 个位置到环形缓存，供 decode 使用 |
| 已提交全局槽 | `[b_max,N_capacity,512]` | 容量按 max_seq_len/ratio 分配，只读取已完成组的前缀 |
| 未完成压缩组 | value/score 的块内 state | 每步写当前槽；组完成后池化、norm、位置处理、量化、写全局槽 |
| 索引 Key | `[b_max,N_capacity,128]` | 跟随其来源摘要提交；供 Indexer 评分，宽度与主 KV 不同 |
| 位置列表 | `[b,s,K]`，int32 | 窗口列表与全局列表拼接；全局位置需要加当前 KV 布局 offset，不可达项保持 −1 |

当一次 query 位于 token t 时，ratio 为 r 的摘要组只有在组尾不晚于 t 时可见，因此可读槽数为 `floor((t+1)/r)`。未完成组保留在 state 中，不作为完整摘要提前开放。本模型 ratio 2 使用非重叠组；ratio 1 直接投影和归一化，没有未完成组 state。

跨层共享存在时，来源层必须先发布缓存、索引 Key 或选择列表，后续层再读取；复用选择列表意味着复用历史位置，不意味着复用主注意力的 Q 或输出。独立层拥有哪些状态、共享层读取哪些状态，应以本模型前面的层类别表为准。

`concat` 的 offset 随当前 KV 布局而变：prefill 的局部区域宽度是当前 chunk 长度，decode 的局部区域宽度是 128。它是 sparse gather 列表的物理偏移，不是 token 的 RoPE 位置。服务系统还可能使用分页缓存和更复杂调度；这里解释的是公开参考实现的连续数组、环形槽与共享 state。

源码入口：[get_window_topk_idxs、Compressor、Indexer 与 Attention 的缓存读写]([本地源码路径])、[sparse_attn 的 int32 位置列表]([本地源码路径])。
## 8. 参数、存储与缓存预算

### 8.1 参数容量与单步计算分开看

以下来自本地原始构造器的 shape 统计：FP4 packed 项按两个逻辑值计数，scale 与整数查表项另记。GB 为十进制，B 参数为十亿；共享 Embedding/Head 只计一次。

| 范围 | 逻辑参数 B | 构造器权重 payload GB（含 scale） |
|---|---:|---:|
| 主干、Embedding/Head、Engram 与视觉组件 | 748.979953264 | 505.049432528 |
| 三个 DSpark stage | 14.225362530 | 8.099641736 |
| 合计 | 763.205315794 | 513.149074264 |

这些数值表示指定参考构造器的权重与 scale 容量；checkpoint 文件封装、设备 allocator、KV、激活、工作区、通信缓冲另计。一个 token 只使用选中的路由专家，全部专家仍占权重存储。

两个 Engram lookup 表合计 196.613849600B 逻辑参数，含投影等的完整 Engram 为 196.928504320B；排除完整 Engram 后主干与视觉等为 552.051448944B。Engram 的容量是稀疏查表参数，不能将它全部当作每 token 的 dense 矩阵计算量。

### 8.2 全局缓存的 890 B/token

生产 FP4 主全局 KV 为 512 维 FP4+32B scale=288B/位置；索引 K 为 128 维 FP4+4B scale=68B/位置。三个 ratio 2 源与一个 ratio 1 源产生的全局位置密度是 3/2+1=2.5：

```text
main KV：288 × 2.5 = 720 B/token
index K：68 × 2.5 = 170 B/token
总 global payload：890 B/token
```

**图 12：全局缓存密度与共享源层**

![全局缓存密度与共享源层]([本地源码路径])

图的紫色与金色条按字节比例显示主 KV 和索引 K。890 只计算指定生产打包格式的全局 payload，局部 SWA、压缩 state、分页尾部、对齐与元数据另计。

参考实现使用 BF16 数组，fake quant 后仍占 BF16 容量：`(512+128)×2×2.5=3200 B/token`。40 层局部窗口另为 `40×128×512×2=5 MiB/序列`；每个请求与容量配置还影响实际分配。

### 8.3 Prefill 与 Decode 的主要工作

Prefill 涉及批量投影、专家矩阵乘法、多 query 的索引/注意力、压缩与缓存写入。Decode 每次处理少量 query，更受权重访问、长历史读取、专家通信与调度影响。主注意力读取数、Indexer 扫描数、权重参数量和实际延迟是不同口径。

CED bounded replay 减少 Decoder 实际处理的提示词 token 数；Engram 增加有限行查表，DSpark 的收益随草稿成本、接受率和服务安排变化。

## 9. 推理框架与设备执行

### 9.1 MegaMoE：MoE 在设备上的执行后端

MegaMoE 是本地 vLLM 快照中的 MoE 执行路径。它接收已选中的专家 ID、路由权重和 token hidden，把专家并行的数据搬运、量化与分组专家矩阵计算等工作交给对应后端；在模型结构中仍位于 MoE 子层，不在 Attention 与 MoE 之间增加新层。

```text
MoE router → topk_ids [T,6]、topk_weights [T,6]
    → 准备激活/scale/路由与对称缓冲
    → 按专家并行执行路由专家：gate/up → SwiGLU → down
    → 加权合并回 token 的 [T,d]
    → 共享专家输出（融合在后端内，或另算后相加）
    → mHC 回写
```

本地 NVIDIA DeepGEMM 路径调用 `prepare_megamoe_inputs` 和 `deep_gemm.fp8_fp4_mega_moe`；框架包装还包含 FlashInfer 的 `stage_inputs/compute` 路径。逻辑专家 ID 可能经过负载平衡映射成物理副本 ID，padding token 也需在路由准备中处理；这些是运行时组织，不改变“按当前 token 选专家再合并”的模型含义。

共享专家是否融合，取决于后端 API、权重/scale 布局、并行方式和开关。未融合时，框架计算 `shared_output` 并加回路由专家输出。不能把“MegaMoE”解释成所有专家都被激活，或把共享专家漏掉。

该 NVIDIA 模型路径要求 expert parallel，并检查 FP4 路由专家、sqrtsoftplus 与选中权重归一化等条件。它们是本文固定框架快照的后端约束；不同平台或其他后端可以采用不同执行路径，普通 MoE 数学结构仍成立。这里说明源码路径与约束，实际 GPU 性能需对应设备和运行配置测量。

源码：[DeepseekV4MoE.forward]([本地源码路径])、[DeepseekV4MegaMoEExperts.forward]([本地源码路径])、[FlashInfer wrapper]([本地源码路径])。

V4.1 在 vLLM 使用独立的 `DeepseekV41ForCausalLM` 多模态 wrapper 与 `DeepseekV41LLMForCausalLM` 主干；它复用 V4 NVIDIA MoE 组件。共享状态、Engram、DSpark 与视觉入口另由 V4.1 模块组织。SGLang 把 nested text/vision 配置规范化后进入含 V4.1 分支的 `DeepseekV4ForCausalLM`，配置类名称不能代替实际数据流。

### 9.2 vLLM V1

```text
API launcher / OpenAI router
  → AsyncLLM：输入处理、请求注册、输出汇聚
  → EngineCoreClient：引擎进程通信
  → EngineCore：执行循环
  → Scheduler.schedule：安排 token 与 KV slots
  → Executor.execute_model：worker 执行
  → Scheduler 更新状态
  → 输出处理与 collector
  → HTTP / 流式响应
```

该快照的 server app/engine 构造入口为 `entrypoints/launchers/api_server/entry.py`。`entrypoints/openai/api_server.py` 是 deprecated compatibility shim。

Scheduler 以 token progress 组织每步执行：已计算 token 数向请求当前 token 数及 speculative token 数推进，同时受 token budget 与 KV slot 预算约束。这种组织方式将 prefill、decode、chunking 与 speculative 执行接入统一的进度管理。

KVCacheManager 与 BlockPool 管理 block 分配、引用和复用。Prefix caching 以完整 token block 的 hash 关联缓存；资源不足时，调度器按策略推迟或抢占请求。EngineCoreClient、Executor 和 worker 提供进程与设备执行边界。

### 9.3 SGLang SRT

```text
FastAPI HTTP / 协议 route
  → TokenizerManager：规范化、tokenize、创建 RID 状态
  → typed request / IPC
  → Scheduler：waiting/running queue、prefix 命中、KV 预算
  → worker / model runner：执行 batch
  → batch result 处理
  → Detokenizer / TokenizerManager：按 RID 回流
  → HTTP / SSE 响应
```

Scheduler 的 normal loop 执行接收、选 batch、运行、处理结果；overlap loop 用队列、stream/event 和依赖控制重叠 CPU 工作与 GPU 执行。新 prefill、running decode 与 chunked request 在循环中衔接。

RadixCache 用 token key 查最长公共前缀，并通过引用保护、插入与淘汰管理状态。命中长度进入 SchedulePolicy 和 PrefillAdder，与缺失 token、chunk 大小、优先级及 KV 预算一起决定准入。实际顺序取决于策略、队列规模和资源条件。

### 9.4 主要抽象边界

| 维度 | vLLM V1 | SGLang SRT |
|---|---|---|
| 前端 | AsyncLLM / EngineClient，按请求汇聚输出 | TokenizerManager、RID 状态与 typed IPC |
| 调度中心 | token progress、显式 token budget | waiting/running batch 与 prefill/decode 衔接 |
| 前缀复用 | block hash、引用与 cache manager | radix longest-prefix 与 admission/policy 联动 |
| 执行隔离 | EngineCore、Executor、worker | Scheduler、worker、model runner |
| 异步组织 | 前端/引擎边界与执行机制 | normal/overlap runtime loop |
| 主要复杂性 | 进程接口、cache groups 与执行边界 | batch 生命周期、cache policy 与 overlap 依赖 |

两个框架都支持前缀缓存、异步执行和分块 prefill。差异主要在抽象边界和状态组织，不能由此推导某个模型的吞吐排名。

**图 13：两种 serving 请求闭环**

![图 13：架构与机制示意]([本地源码路径])

<!-- figure-explanation:11 -->

**读图：请求在哪一步变成执行批次**

两列都从 HTTP 向下到模型执行与输出。左列 vLLM 在 AsyncLLM、EngineCoreClient、EngineCore 和 Executor 间划分职责；右列 SGLang 通过 TokenizerManager、typed IPC、Scheduler 和 worker 组织请求。

**机制解析**

调度器把请求转化为本轮 token 或 batch 工作，并向 KV 管理器申请可用空间。模型结果回到调度状态后再形成 HTTP 输出；图中弯箭头表示结果推动下一轮调度，不表示 HTTP handler 直接运行模型。

vLLM 的 block hash 与 SGLang 的 radix prefix 都参与前缀复用，缓存组织不同并不意味着另一方缺少同类能力。

<!-- /figure-explanation -->


### 9.5 硬件适配

两者都按平台识别、设备能力、运行时后端和算子 dispatch 分层适配。

vLLM 使用 `Platform`、platform plugin、attention backend registry 和按平台组织的线性 kernel 候选。SGLang 使用 `SRTPlatform`、设备接口与运行时工厂，同时保留厂商专用 `hardware_backend`、fused-op 分派和平台条件分支。

平台能被识别只是适配起点。模型能否运行取决于 attention、MoE、量化、KV、graph capture、通信与构建依赖的组合；不同平台可能采用独立 runner。这里的多平台适配与单个平台上的 TP 多卡切分是两件事。

## 10. 训练与后训练

训练设置采用官方报告披露；前向源码不包含完整训练工程。

### 10.1 V4.1 数据与上下文扩展

V4.1 总训练量为 45T token，text-only 与 multimodal token 的混合比例为 7:1。64K 上下文从训练初期使用，在累计 34T token 时扩展至 1M；34T 是扩展节点，不是额外增加的训练量。

多模态预训练主要利用原生网页、PDF 等内容，并处理与文本语料的重叠。数据流程包括图文清洗、alt-text 与图像语义处理、去重和视觉质量筛选，补充 OCR、grounding、代码图像及计算机交互轨迹等领域数据。预训练的原生多模态来源与后训练的合成 Agent 任务需要按阶段理解。[V4.1 技术报告 §4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

### 10.2 V4.1 优化器与 schedule

| 项目 | 报告配置 |
|---|---|
| 矩阵参数 | Muon；momentum 0.95，weight decay 0.1，RMS update 0.18 |
| norm / 非矩阵参数 | AdamW；β=(0.9,0.95)，epsilon `1e-20`，weight decay 0.1 |
| embedding / head | Sinkhorn-balanced 更新；11 次迭代，τ=`1e-3`，epsilon=`1e-20` |
| Engram 学习率 | 相应基准的 5 倍 |
| batch | 固定 100.6M token |
| warmup | 2000 step |
| warmup 后至 28T | 学习率 `2.6e-4` |
| 28T–40T | cosine 衰减至 `2.6e-5` |
| 40T–45T | 维持 `2.6e-5` |
| 34T 节点 | 上下文 64K → 1M |
| 专家负载平衡 | 选择偏置更新率 0.001；小权重序列平衡项 0.0001 |

视觉塔先独立进行 SigLIP 式对比预训练和 autoregressive fine-tuning，再与语言主干集成。报告中的“从头训练”不意味着视觉塔省略独立的前置训练阶段。[V4.1 技术报告 §4.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

**图 14：V4.1 训练阶段与上下文扩展**

![图 14：架构与机制示意]([本地源码路径])

<!-- figure-explanation:12 -->

**读图：学习率曲线与上下文节点怎样对应**

横轴是累计训练 token，纵轴是学习率。蓝线在 warmup 后维持至 28T，在 28–40T 按 cosine 衰减，40–45T 保持低学习率；金色竖线穿过 34T，表示上下文长度在衰减阶段中扩展。

**机制解析**

34T 节点改变的是上下文从 64K 到 1M，不是把预训练总量再增加 34T。底部 SFT→RL→OPD 表示预训练结束后的能力对齐与整合阶段，阶段间箭头不是学习率曲线的一部分。

**28T–40T 的 cosine 衰减**

$$
\eta(T)=2.6\!\times\!10^{-5}+\frac{2.6\!\times\!10^{-4}-2.6\!\times\!10^{-5}}{2}\left[1+\cos\!\left(\pi\frac{T-28}{12}\right)\right]
$$

45T 是累计总量。warmup 的横向宽度被放大用于展示，不按真实 token 比例绘制。

<!-- /figure-explanation -->

### 10.3 V4.1 后训练与训练系统

后训练主流程为 SFT → RL → OPD。重点包括构造任务问题、可交互环境和结果验证器，通过过滤、去重、难度校准和课程组织提高任务质量，再逐步扩大采样计算和执行 scaffolds。

Effort 条件影响推理长度和 token penalty，让模型在同一能力体系下响应不同计算预算。有限训练档位与部署插值一起提供更细粒度的控制。

系统以 sample 为单位异步派发采样任务，减少整组 rollout 的等待。Rollout 与训练可共置、分时使用资源，训练会抢占 rollout；保留 KV 状态支持恢复，并通过跨 checkpoint 的 routing replay 等机制处理样本延续。系统还处理长度偏置、数据并发与 off-policy 约束，避免将吞吐优化与训练分布改变混为一谈。[V4.1 技术报告 §5.1–5.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)



## 11. 算子入口与资料范围

### 11.1 从模块回到源码

| 算子族 | 本文位置 | 本地入口 |
|---|---|---|
| Embedding、布局、RMSNorm、mHC、最终 Head | 第 1、2、7 章 | [主模型]([本地源码路径]) |
| Q/KV、RoPE/YaRN、压缩/状态、Indexer/selector、稀疏 Attention、grouped O | 第 2、7 章 | [模型]([本地源码路径])、[内核]([本地源码路径]) |
| Router、top-k、SwiGLU/clamp、专家合并与残差 | 第 2 章 | [Gate/Expert/MoE]([本地源码路径]) |
| FP8/FP4 量化、scale、GEMM、online softmax、Sinkhorn | 第 7 章 | [Kernel]([本地源码路径]) |
| 并行 collective、缓存 gather/copy/ring、mask/where | 第 2、7 章 | [模型与状态]([本地源码路径]) |
| Engram 的 map、整数 hash、FP8 lookup、copysign gate | 第 4 章 | [Engram]([本地源码路径])、[模型]([本地源码路径]) |
| 图像 resize/pad、patch、2D RoPE、双向 SDPA、SwiGLU、Unfold/GELU | 第 5 章 | [图像处理]([本地源码路径])、[Vision]([本地源码路径]) |
| DSpark、Markov/confidence、采样、EOS/生成控制 | 第 6 章 | [模型]([本地源码路径])、[Generate]([本地源码路径]) |
| MegaMoE、服务打包/调度、分页与前缀缓存、推测验证/回滚 | 第 9 章 | [vLLM]([本地源码路径])、[SGLang]([本地源码路径]) |

这里覆盖公开参考前向的逻辑算子及对应内核作用：输入、布局、投影、归一化、位置、选择、聚合、残差、状态、专家、量化、辅助预测、输出控制与通信。融合 kernel 内的线程分派和设备私有调度按执行方式解释。

### 11.2 配置、报告与固定框架版本

配置：[本模型 inference_config]([本地源码路径])。 [V4.1 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)与[模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md)提供训练和部署口径。

| 框架 | 本文固定快照 |
|---|---|
| vLLM | `c91dccc08044b1269f351ceec13070fe8f075e10` |
| SGLang | `bb9a820f09f97480bfc6d07564fb2b691b8857a3` |
| Transformers（V4） | 5.18.0 |

公式、shape、层类别和状态更新按本地源码解释，参数量按原始构造器统计，训练设置采用报告披露。设备 kernel 数值、完整权重质量、吞吐和多卡峰值内存需要对应权重与 GPU 环境测量。

