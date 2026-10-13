# DeepSeek V4 Pro · 完整算子解析

架构、计算公式、状态更新与推理实现

版本：2026-10-10。按本地固定配置与参考源码解释本模型；从整体架构逐步展开到各模块、公共内核及生成控制。

## 1. 整体架构与层类别

### 1.1 先看输入到输出

文本 token 先经 Embedding，经过 61 个主干 Block，最终收缩、RMSNorm 和词表 Head 给出 logits。每个 Block 依次执行 Attention 与 MoE。模型另有一个 MTP 辅助 Block，读取主干状态与 token 表示。

**图 1：完整模型的主路径与辅助分支**

![完整模型的主路径与辅助分支]([本地源码路径])

最终主干状态先从 `[b,s,4,7168]` 收缩为 `[b,s,7168]`，词表输出宽度为 129,280。主干一次前向和辅助预测有独立调用入口，参数规模与每 token 激活量也应分别理解。

### 1.2 为什么一个 token 有四路残差

Embedding 只产生一个 7168 维向量，随后复制成四份，形状为 `[b,s,4,7168]`。它们属于同一个序列位置的内部状态，初始相同，之后通过不同的混合和回写权重逐渐形成不同表示，没有预先指定的四种语义分工。

每个 Attention/MoE 子层先加权收缩到单路 7168 维，再计算一次子层，最后把输出注入四路并混合旧残差。四路提供可学习的信息保留路径，主 Attention/MoE 的计算与参数并不直接乘四。`hc_mult=4` 是本配置的架构选择。

**图 2：一个标准 Block：Attention、MoE 与四路残差**

![一个标准 Block：Attention、MoE 与四路残差]([本地源码路径])

普通 Block 的执行顺序是：收缩 → Attention RMSNorm → Attention → mHC 回写 → 收缩 → FFN RMSNorm → MoE → mHC 回写。输入输出都是四路状态。 每个子层在自身入口生成并使用自己的 pre/post/comb。

### 1.3 特殊 Attention Block 的类别

**图 3：注意力分支与层职责**

![注意力分支与层职责]([本地源码路径])

SWA 直接读取最近 128 个原始位置；CSA 在局部窗口外加入 ratio 4 的重叠压缩历史，由 Attention 内的 Indexer 选最多 1024 个全局位置；HCA 加入 ratio 128 的非重叠历史，并全读已完成高压缩槽。每种 Block 后续都有 MoE。

| 类别 | 主干层号 | 数量 |
|---|---|---:|
| 纯 SWA | 主干无此类别 | 0 |
| CSA | 2、4、…、60 | 30 |
| HCA | 0、1，加 3、5、…、59 | 31 |

所有层都有局部 SWA 分支。MTP 辅助层号为 61，它的 ratio 为 0；`compress_ratios` 的 62 项包含辅助层。

### 1.4 本模型的关键配置

| 配置 | 数值 |
|---|---:|
| 主干宽度 / 层数 | 7168 / 61 |
| 主注意力头数 / 每头宽度 | 128 / 512 |
| 非 RoPE / RoPE 宽度 | 448 / 64 |
| Q 低秩宽度 | 1536 |
| O 分组 / 每组低秩宽度 | 16 / 1024 |
| 专家总数 / 每 token 路由专家 / 共享专家 | 384 / 6 / 1 |
| 专家中间宽度 / 路由缩放 | 3072 / 2.5 |
| Indexer 头数 / 每头宽度 / top-k | 64 / 128 / 1024 |
| 窗口 / mHC 路数 / Sinkhorn 轮数 | 128 / 4 / 20 |
| 语言 RMSNorm ε | 1e-6 |
| 词表大小 | 129280 |

上下文扩展基准为 65,536，YaRN factor=16，对应 1,048,576 的模型配置。运行时 `max_seq_len` 是单独的缓存容量，参考默认值 4096，需要按使用长度配置。后面的预算按数组或指定生产打包格式计算。

### 1.5 算子阅读路径

先沿第 2 章追踪语言 Block 内的输入、mHC、Q/KV、压缩/共享、Indexer/selector、注意力聚合、O、残差回写、路由和 SwiGLU；再读辅助分支与公共内核。第 7 章展开 RMSNorm、位置变换、量化/GEMM、online softmax、通信和缓存状态，最终覆盖 logits、采样与 EOS 控制。

## 2. 语言主干：逐步展开核心运算

### 2.1 记号与输入

`b` 为 batch，`s` 为本次输入长度，`S` 为已缓存历史与当前输入的总长度，`d` 为隐藏维度，`h` 为注意力头数，`r_q` 为 Q 低秩维度，`g` 为 O 分组数，`r_o` 为每组 O 低秩维度。形状以未切分的官方参考模型为准；生产框架通常将不同请求的 token 打包为总长度 `T`，并按 tensor parallel 切分头和权重。

Embedding 将 `[b,s]` token IDs 映射为 `[b,s,d]`，随后复制为 `[b,s,4,d]`。各层持续保留四路残差；进入注意力或 MoE 时才收缩为 `[b,s,d]`。

### 2.2 mHC：四路残差的受约束混合

每个子层通过 mHC 生成三组系数：

| 系数 | 形状 | 用途 |
|---|---|---|
| `pre` | `[b,s,4]` | 加权收缩四路残差，得到子层输入 |
| `post` | `[b,s,4]` | 将子层输出注入四路残差 |
| `comb` | `[b,s,4,4]` | 混合原有四路残差 |

系数投影先将残差展平为 `[b,s,4d]`。其权重形状为 `[24,4d]`，因为 `(2+4)×4=24`，分别对应 4 个 pre、4 个 post 和 16 个 comb 系数。`comb` 经 20 轮 Sinkhorn 归一化形成近似双随机矩阵，约束跨副本的信息传播。

收缩和回写可表达为：

```text
子层输入 = Σ_i pre_i × residual_i
新残差_i = post_i × 子层输出 + Σ_j comb_ji × residual_j
```

V4 的注意力和 MoE 子层分别计算自己的混合系数。四路残差不是四套独立专家或四次模型前向，而是同一 token 的内部残差状态。

**图 4：mHC 子层中的收缩、执行与回写**

![图 4：架构与机制示意]([本地源码路径])

<!-- figure-explanation:2 -->

**读图：mHC 的计算如何对应这张图**

左侧 x₀…x₃ 是四个 d 维残差向量。蓝色 pre 将它们加权合并，中央 F 执行归一化后的 Attention 或 MoE。绿色 post 将同一个子层输出按不同权重注入右侧四路；下方紫色 comb 同时搬运和混合旧残差。

**机制解析**

下面的 h 对应蓝色汇合点，u 对应 F(x) 的输出，X′ᵢ 对应右侧 yᵢ。回写的两个加数分别来自绿色子层分支和紫色残差分支；即使子层输出很小，残差混合仍然存在。

**① 蓝色汇合：pre 收缩**

$$
\color{#58C4DD}{h}=\sum_{j=0}^{3}\color{#58C4DD}{p_j}\,X_j
$$

**② 中央 F：归一化后执行子层**

$$
\color{#83C167}{u}=F\!\left(\operatorname{RMSNorm}(h)\right),\quad F\in\{\operatorname{Attention},\operatorname{MoE}\}
$$

**③ 右侧回写：新输出与旧残差相加**

$$
X_i^{\prime}=\underbrace{\color{#83C167}{q_i\,u}}_{\text{post injection}}+\underbrace{\sum_{j=0}^{3}\color{#B18AC7}{B_{ji}}\,X_j}_{\text{residual mixing}}
$$

**④ 三组系数来自归一化残差的投影**

$$
\begin{aligned}z&=\frac{W_m\operatorname{vec}(X)}{\sqrt{\operatorname{mean}(X^2)+\epsilon_{\rm norm}}}\in\mathbb{R}^{24}\\p&=\sigma(s_pz_p+b_p)+\epsilon_{\rm hc}\\q&=2\sigma(s_qz_q+b_q)\end{aligned}
$$

**⑤ 紫色矩阵：20 轮 Sinkhorn 约束**

$$
\begin{aligned}B&=\operatorname{Sinkhorn}_{20}(s_Bz_B+b_B),\quad B\in\mathbb{R}^{4\times4}\\B_{ji}&\ge0,\quad \sum_i B_{ji}\approx1,\quad\sum_j B_{ji}\approx1\end{aligned}
$$

以上对单个 token 省略 b、s 下标。p 与 q 分别是 pre、post，B 是 comb；B 的第 j 行第 i 列表示旧副本 j 对新副本 i 的贡献。本模型各子层使用自己生成的 pre。 其中 X 为 [4,d]，h、u 为 [d]，p、q 为 [4]；24 维 z 拆成 4 个 pre、4 个 post 和 16 个 comb 分量，comb 分量重排为 [4,4]。Sinkhorn 先做行 softmax 加稳定项，再交替归一化列与行。

<!-- /figure-explanation -->


### 2.3 Q、KV 与局部注意力

本模型的具体逻辑矩阵为 `wq_a [1536,7168]`、`wq_b [65536,1536]`、`wkv [512,7168]`。O 有 16 组，每组 8 个头；`wo_a` 每组为 `[1024,4096]`，拼接后 `wo_b [7168,16384]`。每个专家的 gate/up 为 `[3072,7168]`，down 为 `[7168,3072]`。

Q 使用两级投影：

```text
x                         [b,s,d]
wq_a(x) → RMSNorm         [b,s,r_q]
wq_b(...)                 [b,s,h×512]
reshape                   [b,s,h,512]
逐头 Q RMSNorm → RoPE     [b,s,h,512]
```

RoPE 作用于最后 64 维，前 448 维不旋转。第一段 Q 的 RMSNorm 与逐头 Q RMSNorm 是两次不同的归一化。

KV 由 `wkv` 生成共享 latent：

```text
RMSNorm(wkv(x))           [b,s,512]
```

这个 latent 同时承担注意力中的 key 和 value 信息，各 Q 头共享它。SWA 保存最近 128 个位置，decode 时写入 `position % 128` 的环形槽位。逻辑位置和物理槽位由注意力下标关联，不能直接把槽位编号当作绝对位置。



V4 的 KV 格式将非 RoPE 的 448 维量化为 FP8，RoPE 的 64 维保留 BF16。参考模型可以用 BF16 数组承载量化后再还原的值；生产缓存则可直接存储打包表示。

### 2.4 SWA、CSA、HCA：同一查询怎样读取历史

下面讨论 DeepSeek V4 Pro 所用的注意力机制。它们共享 Q、局部 KV、attention sink 与 O 投影的主流程，区别在于额外的全局状态怎样生成、怎样选择。动态图演示逐 token 的 decode 状态变化；prefill 则批量处理输入，并为每个 query 施加对应的因果可见范围。

在这三类中，每个主注意力头都是 512 维，其中 64 维应用 RoPE。参考 `Attention` 对纯 SWA 使用 theta=10,000 并关闭 YaRN；对正 ratio 的 CSA/HCA 使用压缩分支的 theta=160,000 与上下文扩展配置，同一层的 Q、局部 KV、全局 KV 共用该层的位置编码设置。不能将 CSA/HCA 的局部路径误解为仍单独沿用纯 SWA 的 RoPE 配置。

**SWA：所有层都有的局部分支**

Pro 主干没有纯 SWA Block；下面的 ratio 0 演示说明通用局部机制及 MTP 的注意力。

SWA（Sliding Window Attention，滑动窗口注意力）的 `compress_ratio=0`。令当前 query 的位置为 t、窗口 W=128，它直接读取的原始位置集合为：

```text
L_t = {max(0, t−W+1), …, t}
|L_t| = min(W, t+1)
```

这里包括当前 token。每个位置的隐藏向量经 `wkv → RMSNorm → RoPE` 生成一个 512 维共享 latent，各查询头读取同一份 KV；最近的位置保持逐 token 表示，没有经过全局压缩。Decode 把新 KV 写入 `t % W` 的环形槽，因此容量固定，旧物理槽会被覆盖。读取时根据逻辑位置构造下标，不能按槽号 0、1、2……直接解释历史顺序。Prefill 的当前输入 KV 可以按完整输入暂存参与计算，但每个 query 的 mask 仍只允许读自己的局部窗口。

纯 SWA 没有 compressor 和 learned Indexer。它限制的是该层对历史 KV 的直接访问范围；多层隐藏态仍可传递更早信息，窗口外历史并非在整个模型中立即消失。

<!-- attention-motion:swa -->

**CSA：细粒度压缩历史，再按 query 选择**

CSA（Compressed Sparse Attention，压缩稀疏注意力）在 SWA 外增加 ratio 4 全局分支。压缩器从该子层的输入隐藏态生成内容向量和逐维 gate，经加权池化形成 512 维主 KV；它并非把已经生成的局部 KV 做固定平均。

把原始序列按 4 个 token 划成块 B₀、B₁……，全局槽 Cⱼ 在块 Bⱼ 完成时提交。Cⱼ 使用上一块 Bⱼ₋₁ 的“重叠半支”和当前块 Bⱼ 的“当前半支”，合计最多 8 个 token 的投影信息；第一个块没有上一块，缺失部分通过无效 gate 排除。每个 token 的投影宽度为 2×512，两半各服务不同的块角色，不能把它理解为同一组 512 维向量被机械重复使用。

```text
C₀ ← B₀                         完成位置 3，位置编码锚点 0
C₁ ← B₀ 的重叠半支 + B₁ 的当前半支  完成位置 7，锚点 4
C₂ ← B₁ 的重叠半支 + B₂ 的当前半支  完成位置 11，锚点 8
```

压缩前 gate 加入块内可学习位置项，沿块内 token 轴、对每个特征维分别 softmax，再加权求和；之后执行 RMSNorm、压缩位置 RoPE 和量化。对已组装好的有效贡献集合 Uⱼ，逐维池化可写成：

```text
v_i = 内容投影(x_i)
g_i = gate 投影(x_i) + 块内位置项
α_i,c = exp(g_i,c) / Σ[l ∈ U_j] exp(g_l,c)
c_j,c = Σ[i ∈ U_j] α_i,c × v_i,c
全局 latent = RoPE(RMSNorm(c_j))，随后按对应格式量化
```

c 是特征维编号；每个维度各自得到一组块内权重。CSA 的 Uⱼ 包含上一块和当前块各自对应半支的贡献，HCA 的 Uⱼ 只包含当前 128-token 块。这种压缩不同于主注意力按 query 对历史位置求 softmax。

只有完成的块会成为全局 KV，未完成块保留在 compressor state 中，不能提前当作摘要读取。位置 t 已完成的全局槽数为 `floor((t+1)/4)`。

独立 Indexer 使用 128 维索引特征：自己的重叠压缩器生成 Indexer K，查询由低秩 Q 特征投影而来。其打分可概括为：

```text
s_tj = Σ_h α_th × ReLU(q_index[t,h] · k_index[j])
I_t  = TopK(s_tj, 已完成且因果可见的全局位置)
A_t  = L_t ∪ {C_j : j ∈ I_t}
```

`α_th` 由当前隐藏态投影产生，包含参考实现的维度与头数缩放；它不是主注意力的 softmax 概率。本模型的全局 top-k 上限为 1024。Indexer 只决定读哪些主 KV，最终 value 聚合使用 512 维主注意力查询与 latent，不能用 Indexer 的 128 维打分直接代替。没有被选中的压缩槽仍留在缓存中，可以被下一 query 选中。

<!-- attention-motion:csa -->

**HCA：更粗的压缩，全读已完成摘要**

HCA（Heavily Compressed Attention，高压缩注意力）也保留 SWA，但全局分支的 ratio 为 128。每个 token 的内容/gate 投影宽度为 512，压缩块互不重叠；逐维门控池化、归一化和位置编码仍参与摘要生成。

```text
C₀ ← token 0–127      到位置 127 才提交，位置编码锚点 0
C₁ ← token 128–255    到位置 255 才提交，位置编码锚点 128
G_t = {C₀, …, C_(floor((t+1)/128)−1)}
A_t = L_t ∪ G_t
```

HCA 不设置 learned Indexer，不做 CSA 式的 query-dependent top-k；本次主注意力读取全部已完成全局槽。它减少的是全局位置的密度，读取数量仍随上下文长度增长，不能说长历史部分具有固定成本。尚未完成的当前 128-token 块保存在 state 中；该块的近期原始 token 仍可以经局部窗口被读取。

<!-- attention-motion:hca -->

**两条历史路径怎样合并？** 局部原始位置与全局压缩位置一起进入同一次主注意力打分、softmax 分母和 value 聚合，再经过逆 RoPE 与分组 O 投影。不是先做两个独立 softmax，再将结果简单相加。全局摘要可能包含局部窗口内 token 的投影信息；它是另一个压缩表示，不要求在原始 token 覆盖范围上与局部窗口完全去重。Attention sink 也进入这个分母，但不提供 value。

以 t=4095、W=128 为例，忽略 padding 槽而按可见记录计数：

| 类型 | 局部原始位置 | 已完成全局槽 | 实际读入的全局槽 | 主注意力读入记录总数 |
|---|---:|---:|---:|---:|
| SWA | 128 | 0 | 0 | 128 |
| CSA / Pro | 128 | 1024 | 最多 1024 | 最多 1152 |
| HCA | 128 | 32 | 32 | 160 |

上述数量是主注意力读取预算，不包括 Indexer 的扫描/打分、压缩器计算、写缓存和投影成本。CSA 的主注意力全局读取有 top-k 上限，但 Indexer 仍需选择候选；HCA 的摘要更少、粒度更粗，不能仅凭表中记录数断言它的质量或整层运行速度更优。

### 2.5 压缩器：ratio 4 与 ratio 128

压缩器对隐藏态做内容投影和 gate 投影，以 gate 权重聚合一个压缩块，随后归一化、施加压缩位置编码并量化。块内 learned positional term 参与门控，不能将其简化成固定平均池化。

| 分支 | 池化结构 | 投影特征宽度 | 全局位置数 |
|---|---|---:|---:|
| ratio 4 | 相邻两个 4-token 块重叠聚合 | `2×512` | 约 `S/4` |
| ratio 128 | 单个 128-token 块聚合，无重叠 | `512` | 约 `S/128` |

例如，一个已有 4096 个位置的 Pro 序列，ratio 4 分支形成约 1024 个全局位置，Indexer 最多选 1024 个；ratio 128 分支形成约 32 个全局位置，主注意力读取全部已完成位置。两类分支都另外保留最近 128 个原始位置。这里是完整历史的数量示例，逐个 query 的实际可见数仍受因果位置与已完成块数限制。

ratio 4 的重叠让新压缩位置同时利用前一块和当前块的信息。ratio 128 保留更粗的长距离摘要，历史位置更少。

Prefill 按完整块生成结果，末尾未完成块保留在 compressor state 中。Decode 每写入一个 token 更新状态，到 `(start_pos+1) % ratio == 0` 时提交新压缩位置。该位置的 RoPE 以块起点 `start_pos+1-ratio` 为基准。缓存中已完成的全局 KV 与尚未完成块的 state 是两类不同状态。

### 2.6 Indexer 与稀疏注意力

Indexer 是 Attention 内部的一条轻量索引分支，selector 是其选择历史位置的阶段。它先筛选全局位置，再让主注意力读取这些位置的 512 维 KV；不是 Attention 后面的另一个模型层。纯 SWA 不需要它，HCA 按完成块直接构造全读索引，V4 的 CSA 才执行 learned Indexer。

**索引 Q、K 与头权重从哪里来？** 对 Attention 入口的归一化隐藏态 x，先计算主 Q 的低秩特征 `qr=RMSNorm(wq_a(x))`。主 Q 和索引 Q 分别使用自己的 `wq_b`；索引 Q 的输出是每头 128 维，本模型有 64 个索引头。V4 索引 K 由独立的 ratio 4 重叠压缩器产生，特征维为 128、块内投影宽度为 256；主 KV 压缩器则输出 512 维，它们参数与缓存分开。

```text
索引 Q = FP4模拟(Hadamard(RoPE(wq_b_index(qr))))  [b,s,H_i,128]
索引 K = FP4模拟(Hadamard(RoPE(索引压缩器(x))))   [b,N,128]
α = weights_proj(x) / √(128 × H_i)               [b,s,H_i]
D[b,t,h,j] = dot(Q_index[b,t,h], K_index[b,j])      [b,s,H_i,N]
S[b,t,j] = Σ_h α[b,t,h] × max(D[b,t,h,j],0)        [b,s,N]
```

这里 `H_i=64`、N 为缓存中完成的压缩位置数；按 tensor parallel 切分头时，各设备算自己的头贡献，再对 S 做 all-reduce。ReLU 只作用于 QK 点积，α 来自线性投影，没有 softmax/sigmoid，也不保证非负；它是索引打分的头权重，不能解释成注意力概率。

**Selector 怎样把打分变成下标？** 对当前绝对位置 t，只允许 `j < floor((t+1)/4)`；prefill 中 batch 化的 score 张量可能包含较晚 query 才能使用的槽，先把当前 query 不可见的 S 置为 `−∞`，再取 top-k。有效索引上限是 `min(配置 top-k, 当前 query 完成的槽数)`；不足部分或无效选择以 `−1` 表示，实际 kernel 会忽略它。

```text
I_t = TopK(S_t 经因果 mask 后的值)
物理全局下标 = I_t + offset
主注意力下标 = concat(局部窗口下标, 物理全局下标)
```

prefill 的 offset 对应当前局部 KV 输入区长度；decode 的 offset 为固定局部池长度 128。Top-k 选择的是“槽的位置”，不能直接使用槽内最大分数作为主注意力权重。主注意力会重新用自己的 512 维 Q/K 打分、softmax、聚合 value。

**图 5：Indexer 的逐头评分、因果 mask 与 selector**

![Indexer 计算]([本地源码路径])

教学例中两头原始点积分别是 `[-2,3,1,4]` 和 `[4,-1,2,1]`。ReLU 后按 α=(0.6,0.4) 汇总得 `[1.6,1.8,1.4,2.8]`；在 t=11，C₃ 尚未完成，mask 后变成 `[1.6,1.8,1.4,−∞]`。演示 top-2 因此选 C₁、C₀。本模型实际 k=1024，这里用 2 是为了展示选择过程。

ratio 4 层有独立 Indexer：Q 投影成多头 128 维查询，K 由自己的压缩器产生；K 压缩器的重叠投影宽度为 256。索引路径使用 Hadamard 变换及低比特表示，以较小特征判断哪些压缩位置值得主注意力读取。

Indexer 对每头 QK 分数进行 ReLU，再按 query-dependent 权重汇总，选取全局 top-k。有效数量受已完成压缩块数限制：

```text
有效 top-k ≤ min(index_topk, 已完成压缩位置数)
```

因果掩码限制可见历史，无效项以 `-1` 标识。Prefill 和 decode 的 KV 拼接布局不同：prefill 可使用当前局部序列长度作为全局索引偏移，decode 使用固定窗口长度 128。偏移必须和实际缓存布局一致。

ratio 128 层不使用 learned Indexer，而是读取全部已完成的高压缩位置。每层注意力的可见集合都包括 SWA 的局部位置，再加该层选择的全局位置；短距离精确信息与长距离压缩信息共同参与计算。

主注意力使用缩放 QK 分数和 softmax 聚合，还为每个头设置可学习的 attention sink：sink 项加入 softmax 分母，不贡献 value，相当于允许该头保留一部分注意力概率而不读入具体历史内容。

**图 6：V4 注意力的 Q、局部 KV 与全局 KV**

![图 5：架构与机制示意]([本地源码路径])

<!-- figure-explanation:3 -->

**读图：查询怎样合并局部与全局记忆**

先看顶部 Q 投影，再看中部 local 与底部 global。蓝色窗口保存最近 128 个原始位置；紫色槽是长历史的压缩表示，金色槽表示被 Indexer 选中。三条路径在右侧汇合，再进入 O 投影。

**机制解析**

每个 query 用局部位置和选中的全局位置共同计算注意力。ratio 4 层用 learned top-k，ratio 128 层读全部已完成压缩槽；未选中槽仍留在缓存中。图中绿色汇合点概括了 QK 打分、softmax 与 value 聚合。

**选中位置的 attention 聚合**

$$
o_{t,h}=\frac{\sum_{j\in\mathcal A_t}\exp(\beta q_{t,h}^{\top}k_j)\,v_j}{\exp(a_h)+\sum_{j\in\mathcal A_t}\exp(\beta q_{t,h}^{\top}k_j)}
$$

公式中的 t 是当前序列位置，h 是注意力头编号；q、k、v 分别是查询、键和值，o 是该头的聚合输出。Aₜ 是可见的局部与全局位置集合，β=1/√512 是分数缩放，aₕ 是仅进入分母的可学习 attention sink。这里用标准 Q/K/V 记号表达聚合；参考实现的 K/V 共用 512 维 latent。本模型包含低秩 q_norm 和展开后的逐头 Q norm。

<!-- /figure-explanation -->

### 2.7 输出投影

注意力输出先做逆 RoPE，再进行分组低秩 O 投影：

```text
attention output          [b,s,h,512]
按 g 组展开               [b,s,g,h×512/g]
每组 wo_a                 [b,s,g,r_o]
合并                      [b,s,g×r_o]
wo_b                      [b,s,d]
```

`wo_a` 的权重可记为 `[g×r_o,h×512/g]`，计算按组组织；`wo_b` 为 `[d,g×r_o]`。这一结构避免直接使用完整的 `[d,h×512]` 输出矩阵，同时保持组内头信息融合。

### 2.8 MoE 路由与专家

**Attention 之后的完整衔接**：O 投影属于 Attention 的末尾，把多头结果恢复为 `[b,s,d]`。Block 先使用 Attention 子层之前保存的四路 residual 和该子层的 post/comb 做 mHC 回写，得到更新的 `[b,s,4,d]`；再从这份更新状态收缩出单路，经过独立的 `ffn_norm`，送入 MoE。不会把未回写的 Attention 向量直接当成 MoE 的完整输入。

```text
u_attn = Attention(RMSNorm_attn(collapse(X)))      [b,s,d]
Y_i = post_attn[i] × u_attn + Σ_j comb_attn[j,i] X_j
x_ffn = RMSNorm_ffn(collapse_ffn(Y))              [b,s,d]
u_ffn = MoE(x_ffn)                               [b,s,d]
Z_i = post_ffn[i] × u_ffn + Σ_j comb_ffn[j,i] Y_j  [b,s,4,d]
下一层接收 Z
```

两次 RMSNorm 使用各自可学习的权重；两次残差混合也有各自的参数。本模型在各子层入口生成自己的 pre/post/comb，分别用于本次收缩、注入和混合。

**图 7：Attention 到 MoE，再到下一层的执行顺序**

![Attention 后续模块]([本地源码路径])

**MoE 做什么？** Attention 读取不同序列位置；MoE 的 FFN 对每个 token 的当前向量进行特征变换。它不再沿序列轴做 QK 注意力，也不另建历史 KV。路由器根据每个 token 的特征决定哪些专家执行，选中的专家分别做相同类型、不同权重的前馈计算。

路由使用 `sqrtsoftplus`：

```text
score = sqrt(softplus(x × W_gate))
选择分数 = score + selection_bias
选择 6 个专家
weight = 被选专家的原始 score / 其 score 总和 × route_scale
```

选出的专家输出还要合并，形成一个 d 维的 MoE 子层结果：

```text
E = 当前 token 选中的 6 个路由专家
MoE(x) = shared_expert(x) + Σ[e ∈ E] weight_e × expert_e(x)
```

每个专家接收同一个收缩后的 x，分别完成自己的 gate/up/down 投影；共享专家的输出单独相加。得到的 MoE(x) 随后经 mHC 回写到四路残差。

选择偏置只影响选中谁，最终混合权重取原始 score。因此路由不是 softmax；六个路由权重之和近似为本模型的 2.5。每个 token 还执行一个共享专家。

V4 前三层使用 token ID 到专家 ID 的 hash 路由表 `tid2eid`，形状为 `[129280,6]`，类型为 int32。它固定专家选择，混合权重仍由当前隐藏态的路由分数生成。后续层按分数 top-k 选择。

单个专家包含 SwiGLU 的 gate/up 两个输入投影和一个输出投影，逻辑权重形状为 `[I,d]`、`[I,d]`、`[d,I]`，其中 `I` 为专家中间维度。`swiglu_limit=10` 用于控制激活范围。大多数专家虽不参与一个 token 的计算，仍占据模型权重存储。

**单个专家内部的算子**：先做两次 `d → I` 的线性投影，gate 支和 up 支经过范围控制，再执行 SiLU 与逐元素乘法，最后 `I → d` 投影。这叫 SwiGLU，不是仅做一次线性变换。

```text
gate = W1 x                       [I]
up   = W3 x                       [I]
gate = min(gate, 10)
up   = clip(up, −10, 10)
a = SiLU(gate) × up                [I]
expert(x) = W2 a                  [d]
SiLU(z) = z × sigmoid(z)
```

这里给出 `swiglu_limit=10` 的参考路径；gate 只限制上界，up 同时限制上下界，不能写成两支都对称 clip。路由标量可在 down 投影前乘入 a，等价于对该专家输出加权。本模型 I=3072；专家输出恢复为 d=7168。

**两个 top-k 的区别**：Indexer 在 Attention 内选历史位置；MoE router 在 FFN 内选专家。前者的输出是历史下标，后者同时输出专家 ID 和专家混合权重。每层每 token 的 6 个路由专家与 1 个共享专家共同形成一个 d 维结果，再进入第二次 mHC 回写。MegaMoE 负责这一专家执行路径，具体见第 6.1 节。

### 2.9 最终收缩与词表 Head

主干最后一层回写后仍有 `[b,s,4,7168]` 状态。主 Head 拥有独立的 mHC 收缩参数：将四路展平后计算 RMS 投影，得到四个 sigmoid 系数（加 1e−6），加权收缩成单路；随后最终 RMSNorm 和词表矩阵计算 logits。

```text
a = sigmoid(scale × W_head vec(X) / sqrt(mean(vec(X)²)+1e−6) + base) + 1e−6
h = Σ_i a_i X_i                         [b,s,7168]
logits = W_vocab RMSNorm(h)              [b,s,129280]
普通生成只输出最后位置                    [b,129280]
```

`W_head` 逻辑形状为 `[4,28672]`；`W_vocab` 为 `[129280,7168]`。四个收缩系数没有 softmax，也不保证和为 1；这套主 Head 参数独立于普通 Block 与 MTP 的 HC-head。词表权重在 checkpoint 中为 BF16，参考 Head 参数以 FP32 保存并输出 FP32 logits。

## 3. MTP：辅助 Block 与生成控制

### 3.1 四路输入融合与完整辅助计算

本模型构造一个独立 MTP Block，层号 61、ratio 0，宽度 7168。它包含自己的 Attention、mHC、384 个路由专家与一个共享专家，另有输入融合投影、Norm 和专用 HC-head；Embedding 与词表矩阵共享。

$$
Z_i=W_e\operatorname{RMSNorm}_e(E[y])+W_h\operatorname{RMSNorm}_h(X_i),\qquad i=0,1,2,3.
$$

token 支得到 `[b,s,7168]` 后广播到四路，hidden 支沿各路自己的 d 维归一化，融合后的 Z 为 `[b,s,4,7168]`。MTP 用自己的 Block 处理它，再经自己的 HC-head、最终 Norm 和共享词表 Head 输出 logits。MTP 的 Attention 为局部窗口路径，仍含低秩 Q/KV、位置处理、sink、逆 RoPE 与分组 O。

**图 8：MTP 的双输入融合、完整 Block 与输出**

![MTP 的双输入融合、完整 Block 与输出]([本地源码路径])

沿图上、下两支先看 hidden 与 token 的独立 Norm/投影，在中间相加后进入完整辅助 Block。返回词表分数前再次收缩四路。MTP 能提供辅助预测，推测解码还需运行时安排主模型校验、接受/拒绝和缓存回滚。

### 3.2 输出采样与普通生成循环

参考普通生成从主干最后位置 logits 采样；温度不大于 0 时取 argmax，正温度时计算分类概率并用指数竞赛采样：

```text
p = softmax(logits / max(T,1e−5))
e_v ~ Exp(1)
next_id = argmax_v(p_v / e_v)
```

输出路径未额外执行 top-p/top-k 词表筛选。Attention 的历史 top-k 与 MoE 的专家 top-k 各有不同输入和用途。

生成循环用 prompt mask 的 where 保留尚未读完的提示词，写 token 数组、比较 EOS、更新 finished mask；所有序列完成或达到生成长度即停止。普通参考脚本没有调用 MTP draft/verify 闭环。

源码：[MTPBlock 与 Head]([本地源码路径])、[生成入口]([本地源码路径])。

## 4. 参数、存储与缓存预算

### 4.1 参数容量与单步计算分开看

以下来自本地原始构造器的 shape 统计：FP4 packed 项按两个逻辑值计数，scale 与整数查表项另记。GB 为十进制，B 参数为十亿；共享 Embedding/Head 只计一次。

| 范围 | 逻辑参数 B | 构造器权重 payload GB（含 scale） |
|---|---:|---:|
| 主干、Embedding/Head | 1572.997201763 | 858.243815884 |
| 完整 MTP 额外组件 | 25.840145979 | 14.023768236 |
| 合计 | 1598.837347742 | 872.267584120 |

这些数值表示指定参考构造器的权重与 scale 容量；checkpoint 文件封装、设备 allocator、KV、激活、工作区、通信缓冲另计。一个 token 只使用选中的路由专家，全部专家仍占权重存储。

### 4.2 本模型的全局 KV 密度

主全局 KV 的目标生产格式为 `448 FP8 + 64 BF16 + 8B scale/补齐 = 584 B/位置`；128 维索引 K 用 FP4 时为 68B，用 FP8 格式时为 132B。30 个 CSA 源每 4 token 产生一槽，31 个 HCA 源每 128 token 产生一槽：

```text
FP4 index-K：584×(30/4+31/128)+68×30/4 = 5031.44 B/token
FP8 index-K：584×(30/4+31/128)+132×30/4 = 5511.44 B/token
```

两种数字都只计 global payload。局部窗口、未完成压缩块、页尾、对齐与 allocator metadata 另计；584B 是记录格式，分配时页布局可能增加容量。参考 fake quant 在 BF16 数组中模拟数值，不能按上述 packed 数字计算其实际数组大小。

### 4.3 Prefill 与 Decode 的主要工作

Prefill 涉及批量投影、专家矩阵乘法、多 query 的索引/注意力、压缩与缓存写入。Decode 每次处理少量 query，更受权重访问、长历史读取、专家通信与调度影响。主注意力读取数、Indexer 扫描数、权重参数量和实际延迟是不同口径。

CSA 的主注意力有 top-k 读取上限，Indexer 仍要选择历史；HCA 全读高压缩摘要，其数量随上下文增长。MTP 的收益随草稿成本与接受率变化。

## 5. 预训练与后训练

### 5.1 V4 预训练

V4 报告披露预训练语料超过 32T token，覆盖数学、代码、网页、长文档及多语言数据。数据处理包括样本级质量筛选，编码体系包含约 128K 词表、特殊协议 token 与 FIM 等代码任务支持。长上下文结构与低比特执行共同服务于训练、推理效率。[V4 技术报告 §4](https://arxiv.org/html/2606.19348)

### 5.2 V4 后训练

V4 先对不同领域分别开展 SFT/RL，形成专长不同的 teacher，再通过多 teacher、全词表 reverse-KL on-policy distillation（OPD）整合能力。报告用 OPD 替代最终的 mixed-RL 整合阶段，并在后训练对专家权重及 Indexer Q/K 使用 FP4 quantization-aware training，使能力适应实际低比特执行。[V4 技术报告 §5](https://arxiv.org/html/2606.19348)

## 6. 推理框架与设备执行

### 6.1 MegaMoE：MoE 在设备上的执行后端

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

本模型在 vLLM 的 `deepseek_v4` 模块接入 registry，SGLang 的模型入口为 `DeepseekV4ForCausalLM`。服务框架常将多请求 token 打包为 `[T,4,7168]`，positions、slot mapping、block table 与 attention metadata 把逻辑序列对应到物理缓存。TP 切分 heads/O 分组及矩阵权重，EP 组织专家执行。

### 6.2 vLLM V1

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

### 6.3 SGLang SRT

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

### 6.4 主要抽象边界

| 维度 | vLLM V1 | SGLang SRT |
|---|---|---|
| 前端 | AsyncLLM / EngineClient，按请求汇聚输出 | TokenizerManager、RID 状态与 typed IPC |
| 调度中心 | token progress、显式 token budget | waiting/running batch 与 prefill/decode 衔接 |
| 前缀复用 | block hash、引用与 cache manager | radix longest-prefix 与 admission/policy 联动 |
| 执行隔离 | EngineCore、Executor、worker | Scheduler、worker、model runner |
| 异步组织 | 前端/引擎边界与执行机制 | normal/overlap runtime loop |
| 主要复杂性 | 进程接口、cache groups 与执行边界 | batch 生命周期、cache policy 与 overlap 依赖 |

两个框架都支持前缀缓存、异步执行和分块 prefill。差异主要在抽象边界和状态组织，不能由此推导某个模型的吞吐排名。

**图 9：两种 serving 请求闭环**

![图 13：架构与机制示意]([本地源码路径])

<!-- figure-explanation:11 -->

**读图：请求在哪一步变成执行批次**

两列都从 HTTP 向下到模型执行与输出。左列 vLLM 在 AsyncLLM、EngineCoreClient、EngineCore 和 Executor 间划分职责；右列 SGLang 通过 TokenizerManager、typed IPC、Scheduler 和 worker 组织请求。

**机制解析**

调度器把请求转化为本轮 token 或 batch 工作，并向 KV 管理器申请可用空间。模型结果回到调度状态后再形成 HTTP 输出；图中弯箭头表示结果推动下一轮调度，不表示 HTTP handler 直接运行模型。

vLLM 的 block hash 与 SGLang 的 radix prefix 都参与前缀复用，缓存组织不同并不意味着另一方缺少同类能力。

<!-- /figure-explanation -->


### 6.5 硬件适配

两者都按平台识别、设备能力、运行时后端和算子 dispatch 分层适配。

vLLM 使用 `Platform`、platform plugin、attention backend registry 和按平台组织的线性 kernel 候选。SGLang 使用 `SRTPlatform`、设备接口与运行时工厂，同时保留厂商专用 `hardware_backend`、fused-op 分派和平台条件分支。

平台能被识别只是适配起点。模型能否运行取决于 attention、MoE、量化、KV、graph capture、通信与构建依赖的组合；不同平台可能采用独立 runner。这里的多平台适配与单个平台上的 TP 多卡切分是两件事。

### 6.6 Transformers V4

Transformers 5.18.0 的 V4 实现位于 `models/deepseek_v4`，配置、模块化源文件和生成的 modeling 文件分开组织。模型入口采用 `DeepseekV4ForCausalLM`，将 embedding、Block、缓存、MoE 与词表输出接入标准 Transformers 调用方式。

缓存中 `DeepseekV4HCACache` 处理高压缩分支，`DeepseekV4CSACache` 扩展其功能以支持带索引的压缩稀疏分支；SWA 保留窗口状态。旧式 `compress_ratios` 会映射为主干 layer types，并排除辅助层配置。

此版本的 V4 ratio 映射支持 0、4、128，不能直接用 ratio 1、2 加 V4.1 字段构造原生 V4.1 模型。HF 的 `num_hash_layers` 与官方参考的 `n_hash_layers` 等字段需要按各自配置接口解释。

## 7. 公共算子与执行内核

本章沿 DeepSeek V4 Pro 的前向路径解释归一化、四路混合、位置处理、量化矩阵乘法、注意力聚合和状态更新。这里的“算子”指具有明确输入输出和计算作用的逻辑步骤；设备执行时可以把多个步骤融合在同一个 kernel 中。`reshape`、`gather` 等布局或索引步骤结合所属模块说明。

统一记号：b 为 batch，s 为当前输入长度，d 为主干宽度，H 为注意力头数，G 为输出投影组数，r_q/r_o 为 Q/O 的低秩宽度，N 为已提交的全局槽数。主语言注意力的 latent 宽度为 512，RoPE 尾部为 64；四路残差 shape 为 `[b,s,4,d]`。

### 7.1 RMSNorm、逐头标准化与张量布局

RMSNorm 对最后一维计算均方和倒平方根，再乘可学习的逐维权重 γ。输入和输出 shape 相同，不减均值：

```text
r = 1 / sqrt(mean(x², last_dim) + ε_norm)
y = γ × x × r
ε_norm = 1e−6
```

参考路径先转 FP32 做均方、rsqrt 和缩放，再转回输入 dtype。Attention/FFN 入口的 RMSNorm 处理收缩后的 `[b,s,d]`；低秩 Q 的 `q_norm` 处理 `[b,s,r_q]`；KV 的 RMSNorm 处理 `[b,s,512]`。归一化作用于哪个维度，要与当前 tensor 的 shape 一起读。

本模型还在展开后的每个 512 维 Q 头上做无可学习权重的 RMS 标准化，ε=1e−6；它与低秩 q_norm 是两个步骤。

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

Indexer 的 128 维 Q/K 在 RoPE 后执行固定 Hadamard：`x′=H_128 x/√128`，H 的元素为 ±1。旋转后再模拟 FP4 量化；它不是每 token 重新生成随机矩阵，主 512 维 Q/KV 不使用这项变换。

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

本模型量化 Linear 的激活沿最后一维每 128 个值一组；FP8 权重每 128×128 一个 scale tile。FP4 专家权重沿每个输出行的 K 方向每 32 个值一组，使用 E8M0 scale。量化 tile、历史压缩组和缓存页具有不同用途。

局部和全局主 KV 的前 448 维每 64 值做 FP8 模拟量化，后 64 维 RoPE 尾部保持 BF16。Indexer 的 128 维 Q/K 在固定 Hadamard 后每 32 值做 FP4，使用 E8M0 scale。

`act_quant(..., inplace=True)` 或 `fp4_act_quant(..., inplace=True)` 在参考路径执行“量化 → 反量化 → 回写输入”，缓存 tensor 仍是 BF16。这种模拟对应低比特数值，不等于将该缓存的物理存储直接变成 FP8/FP4。实际低比特 Linear 权重和 GEMM 输入则有独立量化数组与 scale。

源码入口：[Linear 的权重格式与 Attention/Indexer 中的量化调用]([本地源码路径])、[act_quant、fp4_act_quant 及量化内核]([本地源码路径])。

### 7.5 Linear 与带 scale 的矩阵乘法

参考语言模型的 `linear()` 路径要求 bias=None，逻辑输入 `X[...,K]`，权重 `W[N,K]`，输出 `Y[...,N]`：

```text
Y = X Wᵀ
```

权重 dtype 决定执行普通 `F.linear`、FP8 激活×FP8 权重 GEMM，或 FP8 激活×FP4 权重 GEMM。FP4 专家权重不意味着其输入激活也为 FP4：这条 GEMM 路径先把输入激活量化为 FP8。

量化矩阵乘法需要在对应 K 分组累加时补双方 scale。令量化激活分组宽度 Q=128，两种 GEMM 的 scale 索引分别为：

```text
FP8 激活 × FP8 权重：g 是沿 K 每 Q 个值的分组
Y[m,n] = Σ_g Σ_(k=gQ…(g+1)Q−1)
         A_q[m,k] × W_q[n,k] × s_A[m,g] × s_W[floor(n/Q),g]

FP8 激活 × FP4 权重：g 是沿 K 每 32 个值的子块
Y[m,n] = Σ_g Σ_(k=32g…32(g+1)−1)
         A_q[m,k] × W_q[n,k] × s_A[m,floor(32g/Q)] × s_W[n,g]
```

FP8 权重的 scale 共享于 Q 个输出行和 Q 个 K 值组成的 tile。FP4 权重每输出行、每 32 个 K 值一个 scale：本模型 Q=128，同一个激活 scale 对应四个 FP4 权重子块。FP4 子块在参考内核中经过 FP32 转为 FP8 后参与 GEMM，再用双方 scale 补偿；不能漏掉其打包和 scale 索引。

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

主词表输出头具有自己的 hc_head_fn/base/scale：四路联合 RMS 投影后 sigmoid+1e−6，得到四个收缩系数。MTP 具有另一套 HC-head 参数。两者都不使用末层 FFN pre 替代这个独立收缩。

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

当一次 query 位于 token t 时，ratio 为 r 的摘要组只有在组尾不晚于 t 时可见，因此可读槽数为 `floor((t+1)/r)`。未完成组保留在 state 中，不作为完整摘要提前开放。支持重叠的压缩路径还保存上一组投影，具体覆盖范围遵循注意力章节的压缩图与公式。

跨层共享存在时，来源层必须先发布缓存、索引 Key 或选择列表，后续层再读取；复用选择列表意味着复用历史位置，不意味着复用主注意力的 Q 或输出。独立层拥有哪些状态、共享层读取哪些状态，应以本模型前面的层类别表为准。

`concat` 的 offset 随当前 KV 布局而变：prefill 的局部区域宽度是当前 chunk 长度，decode 的局部区域宽度是 128。它是 sparse gather 列表的物理偏移，不是 token 的 RoPE 位置。服务系统还可能使用分页缓存和更复杂调度；这里解释的是公开参考实现的连续数组、环形槽与共享 state。

源码入口：[get_window_topk_idxs、Compressor、Indexer 与 Attention 的缓存读写]([本地源码路径])、[sparse_attn 的 int32 位置列表]([本地源码路径])。
## 8. 算子入口与资料范围

### 8.1 从模块回到源码

| 算子族 | 本文位置 | 本地入口 |
|---|---|---|
| Embedding、布局、RMSNorm、mHC、最终 Head | 第 1、2、7 章 | [主模型]([本地源码路径]) |
| Q/KV、RoPE/YaRN、压缩/状态、Indexer/selector、稀疏 Attention、grouped O | 第 2、7 章 | [模型]([本地源码路径])、[内核]([本地源码路径]) |
| Router、top-k/hash、SwiGLU/clamp、专家合并与残差 | 第 2 章 | [Gate/Expert/MoE]([本地源码路径]) |
| FP8/FP4 量化、scale、GEMM、online softmax、Sinkhorn | 第 7 章 | [Kernel]([本地源码路径]) |
| 并行 collective、缓存 gather/copy/ring、mask/where | 第 2、7 章 | [模型与状态]([本地源码路径]) |
| MTP 输入融合、辅助 Block/HC-head、采样、EOS/生成控制 | 第 3 章 | [MTP]([本地源码路径])、[Generate]([本地源码路径]) |
| MegaMoE、服务打包/调度、分页与前缀缓存、推测验证/回滚 | 第 6 章 | [vLLM]([本地源码路径])、[SGLang]([本地源码路径]) |

这里覆盖公开参考前向的逻辑算子及对应内核作用：输入、布局、投影、归一化、位置、选择、聚合、残差、状态、专家、量化、辅助预测、输出控制与通信。融合 kernel 内的线程分派和设备私有调度按执行方式解释。

### 8.2 配置、报告与固定框架版本

配置：[本模型 inference_config]([本地源码路径])。 [V4 技术报告](https://arxiv.org/html/2606.19348)提供预训练与后训练口径。

| 框架 | 本文固定快照 |
|---|---|
| vLLM | `c91dccc08044b1269f351ceec13070fe8f075e10` |
| SGLang | `bb9a820f09f97480bfc6d07564fb2b691b8857a3` |
| Transformers（V4） | 5.18.0 |

公式、shape、层类别和状态更新按本地源码解释，参数量按原始构造器统计，训练设置采用报告披露。设备 kernel 数值、完整权重质量、吞吐和多卡峰值内存需要对应权重与 GPU 环境测量。

