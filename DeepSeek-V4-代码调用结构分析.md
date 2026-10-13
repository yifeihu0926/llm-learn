# DeepSeek-V4 代码调用结构分析

> **2026-10-07 本机复核更新**：核心结构、参数/缓存口径与 CPU 边界用例见 [本机核验与更正](DeepSeek-V4-V4.1-本机核验与更正.md)。旧文中的 `[未检出]` 是写作时的检出状态；当前可用路径以新报告为准。GPU 性能、训练与完整模型质量未在本机复现。

> 本文档分析 DeepSeek-V4 的**配置层 → 模型层 → 算子/内核层**完整调用结构，覆盖 4 个实现来源：
> 官方参考实现、HuggingFace Transformers、vLLM、SGLang。
> 所有结论均带 `文件:行号` 出处，可直接跳转核对。
>
> **延伸阅读**：
> - [DeepSeek-V4-官方资料与训练后训练补充.md](DeepSeek-V4-官方资料与训练后训练补充.md) —— 官方来源核验、V4.1 公开训练摘要、V4 训练信息待核验清单
> - [DeepSeek-V4.1-架构与代码分析.md](DeepSeek-V4.1-架构与代码分析.md) —— **V4.1（2026-09 开源）的架构增量**：
>   CED 编码器-解码器、CSA2 跨层共享、Engram 条件记忆、DSpark 投机解码、视觉塔

---

## 0. 数据来源与版本锁定

分析所用的代码已实际拉取到本机 `~/Desktop/deepseek-v4-analysis/`，**未下载任何权重文件**（safetensors / bin 均未获取）。

| 来源 | 获取方式 | 版本 / commit | 本地路径 |
|---|---|---|---|
| 官方参考实现 + 配置 | HuggingFace `deepseek-ai/DeepSeek-V4-Flash`、`DeepSeek-V4-Pro`、`DeepSeek-V4-Pro-Base` 的 `config.json` 与 `inference/*.py` | main 分支（2026-10-02 抓取） | [configs/](../configs/) |
| Transformers | 本机已安装的 `transformers` | 见第 3 节 | [models/deepseek_v4/]([本地源码路径]) |
| vLLM | `git clone --depth 1 --filter=blob:none --sparse` | `c91dccc08044b1269f351ceec13070fe8f075e10`（2026-10-02） | [repos/vllm/](../repos/vllm/) |
| SGLang | 同上 | `bb9a820f09f97480bfc6d07564fb2b691b8857a3`（2026-10-02） | [repos/sglang/](../repos/sglang/) |

### 0.1 权重下载说明

vLLM / SGLang 使用 **blobless + sparse checkout** 只拉取 DeepSeek-V4 相关代码路径：

```bash
# vLLM
git clone --depth 1 --filter=blob:none --sparse https://github.com/vllm-project/vllm.git
git sparse-checkout set vllm/models/deepseek_v4 vllm/models/deepseek_v41 \
    vllm/transformers_utils/configs vllm/parser vllm/tokenizers vllm/renderers \
    vllm/reasoning vllm/tool_parsers vllm/model_executor

# SGLang
git clone --depth 1 --filter=blob:none --sparse https://github.com/sgl-project/sglang.git
git sparse-checkout set python/sglang/srt/models python/sglang/srt/configs \
    python/sglang/srt/layers/attention python/sglang/srt/mem_cache \
    python/sglang/srt/arg_groups python/sglang/srt/function_call \
    python/sglang/srt/managers python/sglang/kernels/ops
```

HuggingFace 侧只抓 `config.json` / `generation_config.json` / `inference/*.py`，**显式跳过** `*.safetensors`、`model.safetensors.index.json`、`tokenizer.json`。

### 0.2 抓取到的官方文件

| 文件 | 大小 | 说明 |
|---|---|---|
| [Flash_config.json](../configs/Flash_config.json) | 1.7 KB | Flash 版 HF 配置（`architectures: DeepseekV4ForCausalLM`） |
| [Flash_inference_config.json](../configs/Flash_inference_config.json) | 991 B | Flash 版官方推理脚本自用配置（字段名与 HF 版**不同**，见 §1.3） |
| [Pro_config.json](../configs/Pro_config.json) | 1.8 KB | Pro 版 HF 配置 |
| [Pro_inference_config.json](../configs/Pro_inference_config.json) | 1.1 KB | Pro 版官方推理脚本自用配置 |
| [Pro-Base_config.json](../configs/Pro-Base_config.json) | 1.8 KB | Pro-Base 版 HF 配置 |
| [official_inference/flash_model.py](../configs/official_inference/flash_model.py) | 38.6 KB | 官方模型定义（参考实现） |
| [official_inference/flash_kernel.py](../configs/official_inference/flash_kernel.py) | 22.2 KB | 官方 Triton/CUDA 内核 |
| [official_inference/flash_generate.py](../configs/official_inference/flash_generate.py) | 6.3 KB | 官方生成入口 |
| [official_inference/flash_convert.py](../configs/official_inference/flash_convert.py) | 7.1 KB | 官方权重转换脚本 |

> **重要事实**：Flash 与 Pro 的 `inference/model.py`、`inference/kernel.py` **完全相同**（MD5 分别为
> `65b5002561ad7ef3b9e5a29baad17867` 与 `e4d8e272f13515b899ef8b145b736001`）。
> 即官方参考实现是**单一代码、纯配置驱动**——模型形状全部来自 `inference/config.json`。
> 这两份参考推理代码中的形状差异由配置决定；训练、权重与服务后端的差异不能由此推断，这也解释了为什么下文所有实现都写成"配置字段驱动"的形式。

---

## 1. 配置层：DeepSeek-V4 的架构定义

### 1.1 HF `config.json` 全字段（Flash 版）

来源：[Flash_config.json](../configs/Flash_config.json)

| 字段 | Flash | Pro | 归属 | 作用 |
|---|---|---|---|---|
| `architectures` | `DeepseekV4ForCausalLM` | 同 | 注册 | 模型类名，三方框架的注册入口键 |
| `model_type` | `deepseek_v4` | 同 | 注册 | Transformers `AutoConfig` 分派键 |
| `vocab_size` | 129280 | 同 | Embedding | 词表大小 |
| `hidden_size` | 4096 | **7168** | 主干 | 隐藏维度 |
| `num_hidden_layers` | 43 | **61** | 主干 | 主干层数（不含 MTP 层） |
| `num_attention_heads` | 64 | **128** | MLA | 注意力头数 |
| `num_key_value_heads` | 1 | 同 | MLA | MQA/MLA 单 KV 头 |
| `head_dim` | 512 | 同 | MLA | 每头维度（= `qk_nope` 448 + `qk_rope` 64） |
| `qk_rope_head_dim` | 64 | 同 | MLA/RoPE | 携带 RoPE 的那部分头维 |
| `q_lora_rank` | 1024 | **1536** | MLA | Q 低秩压缩秩 |
| `o_groups` | 8 | **16** | MLA | 输出投影分组数（Grouped LoRA） |
| `o_lora_rank` | 1024 | 同 | MLA | 输出投影低秩秩 |
| `hc_mult` | 4 | 同 | Hyper-Connections | 超连接流数倍率 |
| `hc_sinkhorn_iters` | 20 | 同 | Hyper-Connections | Sinkhorn 归一化迭代次数 |
| `hc_eps` | 1e-06 | 同 | Hyper-Connections | Sinkhorn 数值稳定项 |
| `n_routed_experts` | 256 | **384** | MoE | 路由专家总数 |
| `n_shared_experts` | 1 | 同 | MoE | 共享专家数 |
| `num_experts_per_tok` | 6 | 同 | MoE | 每 token 激活专家数 |
| `moe_intermediate_size` | 2048 | **3072** | MoE | 单专家中间维 |
| `scoring_func` | `sqrtsoftplus` | 同 | MoE | 路由打分函数 |
| `routed_scaling_factor` | 1.5 | **2.5** | MoE | 路由输出缩放 |
| `topk_method` | `noaux_tc` | 同 | MoE | Top-K 选择法（无辅助损失） |
| `norm_topk_prob` | true | 同 | MoE | Top-K 概率归一化 |
| `swiglu_limit` | 10.0 | 同 | MoE | SwiGLU 门控截断 |
| `num_hash_layers` | 3 | 同 | Hash 路由 | 前 N 层用哈希路由（代替学习式路由）。**注意键名**：HF `config.json` 用 `num_hash_layers`，官方 `inference/config.json` 与 SGLang 用 `n_hash_layers` |
| `index_n_heads` | 64 | 同 | 稀疏索引器 | Indexer 头数 |
| `index_head_dim` | 128 | 同 | 稀疏索引器 | Indexer 头维 |
| `index_topk` | 512 | **1024** | 稀疏索引器 | 每 query 选中的 KV 块数 |
| `sliding_window` | 128 | 同 | 局部注意力 | 滑窗大小 |
| `compress_ratios` | 见 §1.2 | 见 §1.2 | 压缩器 | **逐层**压缩比数组 |
| `compress_rope_theta` | 160000 | 同 | 压缩器 RoPE | 压缩分支专用 RoPE base |
| `rope_theta` | 10000 | 同 | RoPE | 主干 RoPE base |
| `rope_scaling` | YaRN, factor 16, beta_fast 32, beta_slow 1, original_max 65536 | 同 | RoPE | 长上下文外推 |
| `max_position_embeddings` | 1048576 | 同 | RoPE | 最大位置（1M） |
| `rms_norm_eps` | 1e-06 | 同 | Norm | RMSNorm eps |
| `hidden_act` | `silu` | 同 | FFN | 激活函数 |
| `expert_dtype` | `fp4` | `fp4`（Pro-**Base** 为 `fp8`） | 量化 | 专家权重精度 |
| `quantization_config` | FP8, `e4m3`, `ue8m0`, block 128×128 | 同 | 量化 | 非专家权重 FP8 量化 |
| `torch_dtype` | bfloat16 | 同 | 计算 | 计算精度 |
| `num_nextn_predict_layers` | 1 | 同 | MTP | 多 token 预测层数 |
| `tie_word_embeddings` | false | 同 | Embedding | 不共享词表权重 |

### 1.2 `compress_ratios` 逐层布局（关键结构）

`len(compress_ratios) == num_hidden_layers + num_nextn_predict_layers`（本例 MTP=1），**多出的最后一项对应 MTP/nextn 层**。

**Flash（43 层 + 1）**：

```
层号  0  1 | 2  3  4  5 ... 41 42 | 43
ratio 0  0 | 4 128  4 128 ... 4 128 | 0
      └─无压缩─┘ └── 4/128 交替 21+20 层 ──┘  └MTP┘

统计: ratio=0 → 3 层 (0,1,43)
      ratio=4 → 21 层 (偶数层 2..42)
      ratio=128 → 20 层 (奇数层 3..41)
```

**Pro（61 层 + 1）**：

```
层号  0  1  2  3  4  5 ... 59 60 | 61
ratio 128 128 4 128 4 128 ... 4 128 | 0

统计: ratio=0 → 1 层 (61，即 MTP)
      ratio=4 → 30 层
      ratio=128 → 31 层 (含 0,1)
```

**重要：末项的真实归属 = MTP 层。**
`num_hidden_layers = 43`（主干层 0..42），而 `compress_ratios` 有 44 项。
多出的第 44 项（索引 43）**不是占位符，而是被 MTP 层真实读取的**：

```python
# flash_model.py:789-791
self.mtp = torch.nn.ModuleList()
for layer_id in range(args.n_mtp_layers):          # n_mtp_layers 默认 1
    self.mtp.append(MTPBlock(args.n_layers + layer_id, args))   # layer_id = 43 + 0 = 43
# MTPBlock(:738) 继承 Block(:647) → Block.__init__ → Attention(43, args)
# flash_model.py:453
self.compress_ratio = args.compress_ratios[layer_id]           # → compress_ratios[43] == 0
```

即 MTP 层复用 `Block`/`Attention` 代码，因此**必须**在数组里为它预留一项，
且该项为 `0`（MTP 层不做压缩，退化为纯滑窗）。

三个框架对"这一项归谁管"的处理**各不相同，但各自自洽**：

| 实现 | 第 44 项处理 | 理由 |
|---|---|---|
| 官方 | **读取**（MTP 层经 `Block`→`Attention` 读到） | MTP 复用主干 Block |
| Transformers | **截断丢弃**（`configuration_deepseek_v4.py:276` `self.layer_types[:n]`） | 其 config 注明 MTP "not instantiated here"（`:91-93`） |
| vLLM | **忽略**：`layer_id < num_hidden_layers` 才读数组，否则强制 `1`（`attention.py:232-235`） | 注释："MTP layer is not included in the compress ratio list"；MTP 由 `DeepSeekV4MTP` 单独承担 |

含义：
- `ratio == 0` → 该层**不做压缩**（走纯 MLA / 滑窗路径）；
- `ratio == 4` → 轻度压缩层（重叠窗口 + 学习式索引器）；
- `ratio == 128` → 强压缩层（非重叠窗口 + 固定 topk）。
- Transformers 把这三档显式命名为三种 layer type
  （`configuration_deepseek_v4.py:28-32`）：
  `0 → sliding_attention`、`4 → compressed_sparse_attention (CSA)`、
  `128 → heavily_compressed_attention (HCA)`。
- Flash 与 Pro 的**层内压缩模式相反**：Flash 前两层不压缩，Pro 前两层是强压缩。
  这说明压缩率排布是逐代调优的架构超参，而非固定规律。

### 1.3 官方推理脚本的配置字段映射

官方 `inference/config.json` 用**短名**，与 HF 长名一一对应：

| 官方推理字段 | HF `config.json` 字段 | Flash 值 | Pro 值 |
|---|---|---|---|
| `dim` | `hidden_size` | 4096 | 7168 |
| `n_layers` | `num_hidden_layers` | 43 | 61 |
| `n_heads` | `num_attention_heads` | 64 | 128 |
| `moe_inter_dim` | `moe_intermediate_size` | 2048 | 3072 |
| `n_activated_experts` | `num_experts_per_tok` | 6 | 6 |
| `score_func` | `scoring_func` | `sqrtsoftplus` | `sqrtsoftplus` |
| `route_scale` | `routed_scaling_factor` | 1.5 | 2.5 |
| `rope_head_dim` | `qk_rope_head_dim` | 64 | 64 |
| `window_size` | `sliding_window` | 128 | 128 |
| `original_seq_len` | `rope_scaling.original_max_position_embeddings` | 65536 | 65536 |
| `rope_factor` | `rope_scaling.factor` | 16 | 16 |
| `beta_fast` / `beta_slow` | `rope_scaling.beta_fast` / `.beta_slow` | 32 / 1 | 32 / 1 |
| `dtype` | `quantization_config.fmt` | `fp8` | `fp8` |
| `scale_fmt` | `quantization_config.scale_fmt` | `ue8m0` | `ue8m0` |
| `n_hash_layers` | `n_hash_layers` | 3 | 3 |
| `hc_mult` | `hc_mult` | 4 | 4 |

官方脚本读入后包装成 `ModelArgs` dataclass，再逐字段下发给各子模块
（详见 §2）。

### 1.4 DeepSeek-V4 相对 V3 的架构增量（配置视角）

从字段对比可直接看出 V4 新增的四组配置簇，这也是三方框架适配工作量所在：

1. **Hyper-Connections 簇**：`hc_mult` / `hc_sinkhorn_iters` / `hc_eps`（V3 无）。
2. **压缩器簇**：`compress_ratios`（逐层数组）/ `compress_rope_theta`（V3 无）。
3. **稀疏索引器簇**：`index_n_heads` / `index_head_dim` / `index_topk`（V3 无，
   V3.2 的 DSA 是雏形，V4 变为 `index_topk` 显式配置）。
4. **Hash 路由簇**：`num_hash_layers`（V3 无），前 3 层用哈希路由替代学习式路由。
5. 路由打分从 `softmax` 换为 `sqrtsoftplus`（`scoring_func`）。
6. `o_groups` / `o_lora_rank` 引入分组低秩输出投影（V3 只有 `q_lora_rank`）。

---

## 2. 官方参考实现调用结构（`inference/model.py`）

文件：[official_inference/flash_model.py](../configs/official_inference/flash_model.py)（827 行）+ [flash_kernel.py](../configs/official_inference/flash_kernel.py)

这是**唯一权威的架构定义**——vLLM/SGLang 的适配都是对它的工程化重写。

### 2.1 配置 → 模型的第一跳

```
flash_generate.py:94   args = ModelArgs(**json.load(open(config)))
  └─ flash_model.py:35  class ModelArgs        # dataclass，字段名与 inference/config.json 键名一致
       └─ flash_model.py:772  Transformer.__init__(args)
```

`ModelArgs` 是配置到模型的**唯一入口**，所有子模块通过 `args.<字段>` 读取超参
（`flash_model.py:439` 起）。这解释了 §1.3 的映射表。

### 2.2 完整调用图

```
Transformer.forward(input_ids, start_pos)                       flash_model.py:802
├─ ParallelEmbedding.forward(input_ids)                         :96
├─ h.unsqueeze(2).repeat(1,1,self.hc_mult,1)                    :805   ← HC 展开为 hc_mult 份
├─ for layer in self.layers:   Block.forward(x, start_pos, input_ids)   :688
│  │
│  ├─【注意力子层】
│  ├─ Block.hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)  :673
│  │   └─ hc_split_sinkhorn(...)          → flash_kernel.py:430
│  │        └─ hc_split_sinkhorn_kernel   → flash_kernel.py:372  (Triton)
│  ├─ RMSNorm.forward  (attn_norm)                             :191
│  ├─ Attention.forward(x, start_pos)                          :484
│  │   ├─ Linear.wq_a → RMSNorm.q_norm  (低秩 Q 压缩)           :496
│  │   ├─ ColumnParallelLinear.wq_b → unflatten 成头            :497
│  │   ├─ q *= rsqrt(mean(q²)+eps)   ← 注意：Q 侧再做一次归一化  :498
│  │   ├─ apply_rotary_emb(q[..., -rd:], freqs_cis)            :499 → :232
│  │   ├─ Linear.wkv → RMSNorm.kv_norm                         :502-503
│  │   ├─ apply_rotary_emb(kv[..., -rd:])                      :504
│  │   ├─ act_quant(kv[..., :-rd], 64, ...)  ← FP8 模拟 QAT      :506 → kernel:105
│  │   ├─ get_window_topk_idxs(win, bsz, seqlen, start_pos)    :507 → :255   (lru_cache)
│  │   ├─ if compress_ratio:
│  │   │   ├─ offset = kv.size(1) if start_pos==0 else win     :509
│  │   │   ├─ Indexer.forward(x, qr, start_pos, offset)        :511 → :402   【仅 ratio==4】
│  │   │   │   ├─ ColumnParallelLinear.wq_b(qr) → unflatten    :411-412
│  │   │   │   ├─ apply_rotary_emb(q[..., -rd:])               :413
│  │   │   │   ├─ rotate_activation (Hadamard) → fp4_act_quant  :414-416
│  │   │   │   ├─ Compressor.forward(x, start_pos)  ← 自带压缩器 :417 → :316
│  │   │   │   │   ├─ Linear.wkv / Linear.wgate                :323-324
│  │   │   │   │   ├─ overlap_transform(...)   ← overlap 模式    :307
│  │   │   │   │   ├─ (kv * score.softmax(2)).sum(2)  ← 门控池化 :342
│  │   │   │   │   ├─ RMSNorm / apply_rotary_emb               :362-367
│  │   │   │   │   ├─ rotate_activation (Hadamard)             :369 → :247
│  │   │   │   │   └─ fp4_act_quant                            :370 → kernel:186
│  │   │   │   ├─ weights = weights_proj(x) * (scale * n_heads**-0.5)  :418
│  │   │   │   ├─ index_score = einsum("bshd,btd->bsht", q, kv_cache)  :420
│  │   │   │   ├─ index_score = (relu(index_score) * weights).sum(2)   :421
│  │   │   │   ├─ dist.all_reduce(index_score)   ← TP 归约      :423
│  │   │   │   └─ topk(min(index_topk, end_pos//ratio)) → 下标   :427
│  │   │   └─ else: get_compress_topk_idxs(ratio,...)          :513 → :269  【ratio==128】
│  │   │   └─ topk_idxs = cat([window_idxs, compress_idxs])    :514
│  │   ├─ if start_pos == 0 (prefill):
│  │   │   ├─ 写 kv_cache 环形窗口                              :520-523
│  │   │   ├─ Compressor.forward(x, start_pos)                 :525 → :316
│  │   │   └─ sparse_attn(q, kv, attn_sink, topk_idxs, scale)  :528 → kernel:355
│  │   │        └─ sparse_attn_kernel                          → kernel:277 (Triton)
│  │   └─ else (decode):
│  │       ├─ kv_cache[:bsz, start_pos % win] = kv             :530
│  │       ├─ Compressor.forward(x, start_pos)                 :532
│  │       └─ sparse_attn(q, self.kv_cache[:bsz], ...)         :533
│  ├─ apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)  :534
│  ├─ o.view(bsz, seqlen, n_local_groups, -1)   ← 分组           :537
│  ├─ torch.einsum("bsgd,grd->bsgr", o, wo_a)  ← 分组低秩 O 投影  :541
│  ├─ RowParallelLinear.wo_b                                    :542 → :173
│  ├─ Block.hc_post(x, residual, post, comb)                   :683
│  │
│  ├─【FFN / MoE 子层】
│  ├─ Block.hc_pre(x, hc_ffn_fn, ...)                          :696
│  ├─ RMSNorm.forward (ffn_norm)                               :191
│  ├─ MoE.forward(x, input_ids)                                :629
│  │   ├─ Gate.forward(x, input_ids)                           :632 → :564
│  │   │   ├─ linear(x.float(), weight.float())                :565
│  │   │   ├─ sqrtsoftplus: F.softplus(scores).sqrt()          :571
│  │   │   ├─ scores += self.bias   (仅影响选择，不影响权重)      :575
│  │   │   ├─ if hash: indices = self.tid2eid[input_ids]       :577  ← 哈希路由
│  │   │   │  else:   indices = scores.topk(topk)[1]           :579
│  │   │   ├─ weights = original_scores.gather(1, indices)     :580
│  │   │   ├─ weights /= weights.sum(-1)                       :582
│  │   │   └─ weights *= route_scale                           :583
│  │   ├─ torch.bincount(indices) → 按专家分桶                  :634
│  │   ├─ for i in local_experts: Expert.forward(x[idx], w)    :640 → :596
│  │   │   ├─ gate = w1(x).float(); up = w3(x).float()         :598-599
│  │   │   ├─ clamp(up, ±swiglu_limit); clamp(gate, max=limit) :601-602
│  │   │   ├─ x = F.silu(gate) * up                            :603
│  │   │   └─ w2(x.to(dtype))                                  :606
│  │   ├─ dist.all_reduce(y)                    ← TP 归约        :642
│  │   └─ y += self.shared_experts(x)           ← 共享专家       :643
│  └─ Block.hc_post(x, residual, post, comb)                   :699
│
├─ ParallelHead.forward(h, hc_head_fn, hc_head_scale, hc_head_base, norm)   :718
│   ├─ ParallelHead.hc_head(...)                               :728
│   │   └─ pre = sigmoid(mixes*hc_scale + hc_base) + hc_eps    :733  ← 与 Block 的 Sinkhorn 不同
│   ├─ get_logits(norm(x))  = F.linear(x[:,-1].float(), W)     :715-716
│   └─ dist.all_gather(logits)                                 :722-725
└─

MTPBlock.forward(x, start_pos, input_ids)                      :757
├─ ParallelEmbedding.embed(input_ids) → enorm                  :760-761
├─ hnorm(x) → e_proj(e).unsqueeze(2) + h_proj(x)               :762-763
├─ Block.forward(...)   ← 复用父类                              :764
└─ ParallelHead.head(x, hc_head_fn, ...)                       :765
```

### 2.3 四个 DeepSeek-V4 独有机制（官方实现视角）

#### (1) Hyper-Connections（HC）：`Block.hc_pre` / `Block.hc_post`

官方实现把"残差流"从 1 份扩成 `hc_mult = 4` 份：

- 展开：`Transformer.forward:805` `h.unsqueeze(2).repeat(1,1,hc_mult,1)` → 形状 `[b,s,hc,d]`
- 收缩：`Block.hc_pre:673` 用 **Sinkhorn 归一化**产生三组系数
  `pre / post / comb`，其中 `mix_hc = (2 + hc_mult) * hc_mult = 24`（`:663`）
- 写回：`Block.hc_post:683`
  `y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`
- 尾部 `ParallelHead.hc_head:728` 用 **sigmoid + hc_eps**（不是 Sinkhorn）做最后一次收缩

三组可学习参数：`hc_attn_fn/ffn_fn`（`[24, 4*d]`）、`hc_*_base`（`[24]`）、`hc_*_scale`（`[3]`），
全部 fp32（`:665-671`）。

#### (2) Compressor：逐层不同压缩比

- 压缩比来源：`Attention.__init__:453` `self.compress_ratio = args.compress_ratios[layer_id]`
  —— 这是 `compress_ratios` 数组与层号的**唯一绑定点**。
- 压缩器实例化：`:466-471`
  ```python
  if self.compress_ratio:
      self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
      if self.compress_ratio == 4:
          self.indexer = Indexer(args, self.compress_ratio)   # 4 → 有学习式索引器
      else:
          self.indexer = None                                 # 128 → 固定 topk 下标
  ```
- 压缩块大小：`Compressor.__init__:290` `self.overlap = compress_ratio == 4`
  —— **ratio==4 走重叠窗口**（`coff = 1 + overlap = 2`），ratio==128 走非重叠。
- 门控池化：`:342` `kv = (kv * score.softmax(dim=2)).sum(dim=2)`
- 重叠变换：`Compressor.overlap_transform:307`，把前半维给"跨块"、后半维给"块内"（`:296`）
- 解码期增量状态：`kv_state` / `score_state`（`:303-304`），
  decode 每步写入 `start_pos % ratio` 槽位，攒满即压缩并前移（`:343-359`）
- 缓存尺寸：`Attention.__init__:473`
  `kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)`
  —— **滑窗段 + 压缩段拼接**在一根 KV cache 里。

#### (3) 稀疏注意力 Indexer

- `index_n_heads=64`、`index_head_dim=128`、`index_topk=512/1024`
- `Indexer` 内部**自带一个 Compressor**（`:398`），并且 `rotate=True` → 走 Hadamard 旋转 + FP4 量化（`:368-370`）
- 只有 `compress_ratio == 4` 的层才有 Indexer；`==128` 的层用 `get_compress_topk_idxs` 直接算下标（`:513`）
- 两类下标拼接后送入 `sparse_attn`（`:514`）

#### (4) Hash 路由（前 3 层）

- `Gate.__init__:556` `self.hash = layer_id < args.n_hash_layers`
- hash 层参数不同：持 `tid2eid`（`[vocab_size, n_activated_experts]`，int32）且 `bias = None`（`:558-560`）
- 非 hash 层持 `bias`（`:562`）
- `Gate.forward:576-579`：`if self.hash: indices = self.tid2eid[input_ids] else: scores.topk(...)`
- 打分函数统一为 `sqrtsoftplus`（`:571` `F.softplus(scores).sqrt()`）

### 2.4 一个容易被忽略的关键分支：RoPE 随 compress_ratio 切换

`Attention.__init__:475-482`：

```python
if self.compress_ratio:
    original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta   # 65536 / 160000
else:
    original_seq_len, rope_theta = 0, args.rope_theta                                # 0 / 10000（关掉 YaRN）
freqs_cis = precompute_freqs_cis(rope_head_dim, max_seq_len, original_seq_len,
                                 rope_theta, rope_factor, beta_fast, beta_slow)
```

**结论**：`compress_ratio == 0` 的层（Flash 的 0、1 层）**不使用 YaRN**、用 base `rope_theta=10000`；
有压缩的层使用 YaRN（factor 16）+ `compress_rope_theta=160000`。
即同一模型内不同层使用**不同的位置编码配置**。

### 2.5 内核层（`flash_kernel.py`）

| 内核 | Triton 定义 | Python 封装 | 被谁调用 |
|---|---|---|---|
| 激活 FP8 量化 | `act_quant_kernel:41` | `act_quant:105` | `Attention.forward:506`、`Compressor.forward:372` |
| FP4 激活量化 | `fp4_quant_kernel:129` | `fp4_act_quant:186` | `Compressor.forward:370`（rotate 路径） |
| FP8 GEMM | `fp8_gemm_kernel:204` | `fp8_gemm:257` | 线性层 |
| 稀疏注意力 | `sparse_attn_kernel:277` | `sparse_attn:355` | `Attention.forward:528,533` |
| HC Sinkhorn | `hc_split_sinkhorn_kernel:372` | `hc_split_sinkhorn:430` | `Block.hc_pre:679` |
| FP4 GEMM | `fp4_gemm_kernel:442` | `fp4_gemm:518` | 专家前向（`expert_dtype=fp4`） |

---

## 3. HuggingFace Transformers 调用结构

位置：`transformers 5.18.0`
`~/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/`

| 文件 | 行数 | 角色 |
|---|---|---|
| `configuration_deepseek_v4.py` | 324 | 配置（**含 V4 的关键语义转换**） |
| `modular_deepseek_v4.py` | 1254 | **人工编写源文件**（结构真相） |
| `modeling_deepseek_v4.py` | 1546 | 由 modular 自动生成（勿直接改） |

### 3.1 配置层的三处关键语义转换

`configuration_deepseek_v4.py` 不是简单的字段容器，它把 HF 的**扁平配置**
翻译成 V4 的**逐层类型调度**：

```python
DEEPSEEK_V4_LAYER_TYPES = ("sliding_attention",
                           "compressed_sparse_attention",     # CSA, ratio 4
                           "heavily_compressed_attention")    # HCA, ratio 128     :21-25

_COMPRESS_RATIO_TO_LAYER_TYPE = {0: "sliding_attention",
                                 4: "compressed_sparse_attention",
                                 128: "heavily_compressed_attention"}             :28-32

DEEPSEEK_V4_MLP_LAYER_TYPES = ("hash_moe", "moe")                                 :35
```

`__post_init__`（`:246-321`）依序做四件事：

1. **压缩比 → 层类型**（`:266-276`）
   `compress_ratios` 的整数数组被逐项映射成 `layer_types` 字符串数组，
   再 `self.layer_types[:n]` 截断到 `num_hidden_layers`
   —— 这就是 §1.2 中"第 44 项不会被读"的显式证据。
   若配置里没给，则用 V4-Pro 默认排布：
   `["heavily_compressed_attention"]*2 + 从第 3 层开始的 CSA/HCA 交替`（`:271-275`）。
2. **Hash 路由 → MLP 层类型**（`:278-282`）
   `num_hash_layers`（HF 键名）→ 前 N 层 `"hash_moe"`，其余 `"moe"`。
3. **RoPE 分型**（`:284-292`）
   由 `qk_rope_head_dim / head_dim` 推出 `partial_rotary_factor`。
4. **两套 RoPE 参数**（`:294-321`）
   ```python
   self.rope_parameters = {
       "main":     {"rope_type": "default", "rope_theta": 10000,  "partial_rotary_factor": ...},
       "compress": {**yarn, "rope_theta": 160000, "partial_rotary_factor": ...,
                    "attention_factor": 1.0},   # yarn 时强制 attention_factor=1.0
   }
   ```
   注释（`:294-299`）明确写道：**YaRN 只作用于有 compressor 的层（CSA/HCA）；
   纯滑窗层用 `theta=10000` 且不做 scaling**——与 §2.4 从官方实现读出的结论**完全一致**，
   这是四份实现互相印证的强证据。

### 3.2 调用图

```
DeepseekV4ForCausalLM  (= MixtralForCausalLM，未覆写)            modular:1246
  └─ DeepseekV4Model.forward(...)                                modular:1192  (继承 LlamaModel)
     ├─ DynamicCache(config=self.config)                         :1205
     ├─ self.embed_tokens(input_ids)                             :1207
     ├─ position_ids = arange(...) + past_seen                   :1210
     ├─ create_sliding_window_causal_mask(...)                    :1218
     │    ← V4 没有 _update_causal_mask，用自由函数（masking_utils.py:1106）
     │    ← 若 attention_mask 是 dict（generate() 预建）则直接取第一个 :1215-1216
     ├─ hidden_states.unsqueeze(2).expand(-1,-1,hc_mult,-1)      :1225  ← HC 展开
     ├─ position_embeddings = {"main": rotary_emb(..., "main"),
     │                         "compress": rotary_emb(..., "compress")}   :1226-1229
     ├─ for layer in self.layers: DeepseekV4DecoderLayer.forward  :1231 → :1048
     │   ├─ post, comb, collapsed = self.attn_hc(hidden_states)   :1062 → DeepseekV4HyperConnection:848
     │   │     ├─ input_norm = DeepseekV4UnweightedRMSNorm        :864 → :75
     │   │     ├─ F.linear(flattened, self.fn.float())            :866
     │   │     ├─ pre/post/comb 拆分                              :868-870
     │   │     ├─ pre  = sigmoid(pre_w*pre_scale + pre_b) + hc_eps        :876
     │   │     ├─ post = 2*sigmoid(post_w*post_scale + post_b)            :877
     │   │     ├─ comb = softmax(comb_w*comb_scale + comb_b, -1) + hc_eps :878
     │   │     ├─ Sinkhorn-Knopp 迭代 (hc_sinkhorn_iters=20)      :884-887
     │   │     └─ collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(2) :891
     │   ├─ self.self_attn(self.input_layernorm(collapsed), **kwargs)      :1063
     │   │   └─ DeepseekV4Attention.forward                       :706
     │   │      ├─ cos, sin = position_embeddings[self.rope_layer_type]   :719
     │   │      │     rope_layer_type = "main" if layer_type=="sliding_attention" else "compress"  :682
     │   │      ├─ q_residual = q_a_norm(q_a_proj(hidden_states))         :721
     │   │      ├─ q = q_b_norm(q_b_proj(q_residual))                     :722-723
     │   │      ├─ apply_rotary_pos_emb(q, cos, sin)                      :724 → :46
     │   │      ├─ kv = kv_norm(kv_proj(hidden_states))                   :726
     │   │      ├─ kv = apply_rotary_pos_emb(kv, cos, sin)                :727
     │   │      ├─ past_key_values.update(kv, kv, layer_idx) ← K=V 共享    :730
     │   │      ├─ if self.compressor:
     │   │      │   ├─ DeepseekV4CSACompressor.forward / HCACompressor.forward  :734
     │   │      │   │    → :571 / :342
     │   │      │   │    · store_compression_weights(...)  (缓存层方法)   :181
     │   │      │   │    · update_overlap_state(...)  【CSA only】         :256
     │   │      │   │    · update_compressor_states(...)                  :204
     │   │      │   └─ block_bias（逐 query 因果 + indexer 有效性）         :736
     │   │      ├─ kv = cat([kv, compressed_kv], dim=2)                   :737
     │   │      ├─ attention_mask = cat([mask, block_bias], dim=-1)       :745-749
     │   │      ├─ attention_interface(q, kv, kv, mask, sliding_window,
     │   │      │                    s_aux=self.sinks, ...)               :751-765
     │   │      ├─ apply_rotary_pos_emb(attn_output, cos, -sin) ← 逆旋转    :773
     │   │      ├─ grouped = reshape(..., o_groups, -1)                   :775
     │   │      ├─ o_a_proj(grouped) → DeepseekV4GroupedLinear.forward     :776 → :301
     │   │      └─ o_b_proj(grouped)                                      :777
     │   ├─ hidden_states = post*attn_output + comb.T @ hidden_states      :1064-1066
     │   ├─ post, comb, collapsed = self.ffn_hc(hidden_states)             :1068
     │   ├─ self.mlp(post_attention_layernorm(collapsed), input_ids=...)   :1069
     │   │   └─ DeepseekV4SparseMoeBlock.forward                          :1012
     │   │      ├─ is_hash = mlp_layer_types[layer_idx]=="hash_moe"       :1007
     │   │      ├─ gate = DeepseekV4HashRouter | DeepseekV4TopKRouter      :1008 → :977 / :960
     │   │      │    ├─ [HashRouter] indices = tid2eid[input_ids]         :998
     │   │      │    │   weights = scores.gather(1, indices)              :999
     │   │      │    └─ [TopKRouter] indices = topk(scores + e_score_correction_bias) :971
     │   │      │        weights = scores.gather(1, indices)              :972
     │   │      ├─ DeepseekV4Experts.forward(flat, indices, weights)      :1020 → :942
     │   │      │    · _apply_gate(gate_up) → SwiGLU + swiglu_limit       :933
     │   │      └─ routed + DeepseekV4MLP(residual)  ← 共享专家            :1021
     │   └─ hidden_states = post*mlp_output + comb.T @ hidden_states       :1070-1072
     └─ self.norm(self.hc_head(hidden_states))                    :1242
          └─ DeepseekV4HyperHead.forward                          :907
```

### 3.3 三种缓存（V4 最核心的工程结构）

```
DeepseekV4HCACache(DynamicSlidingWindowLayer)                    modular:134
   _layer_type = "heavily_compressed_attention"                  :157
   compress_rate = config.compress_rates["heavily_compressed_attention"]  :161
   状态（全部按 name 做 dict，HCA 只用 "compressor"）：
     buffer_kv["compressor"]   / buffer_gate["compressor"]   ← 未满一窗的源 token
     compressed_kv["compressor"]                              ← 已产出的压缩条目
     entry_count["compressor"]                                ← 已产出条数
   方法：
     update(k, v)                     :167  滑窗 K=V 环形更新（keys 即 values）
     store_compression_weights(...)   :181  拼接 + 切出窗口对齐前缀 + 余量留 buffer
     update_compressor_states(...)    :204  追加压缩条目 + entry_count += 1
     reset()                          :217

DeepseekV4CSACache(DeepseekV4HCACache)                           modular:225
   _layer_type = "compressed_sparse_attention"                   :244
   compress_rate = config.compress_rates["compressed_sparse_attention"]  :248
   额外给 buffer_kv/buffer_gate/compressed_kv/entry_count 加 "indexer" 键  :249-252
   额外 overlap_kv / overlap_gate（仅 CSA）                       :253-254
     update_overlap_state(...)       :256  ← 只保留上一窗的 Ca 半（:head_dim）
   reset()                           :272
```

**CSA 的"重叠"语义**（`:231-236`）：`kv_proj`/`gate_proj` 每个源 token 产出 `2*head_dim`
特征 —— 两个独立压缩序列 Ca（`[..., :head_dim]`）与 Cb（`[..., head_dim:]`）。
第 w 个池化条目 = 第 w-1 窗的 Ca 与第 w 窗的 Cb 的 softmax 门控凸组合，
**有效宽度 2*compress_rate，步长 compress_rate**。
跨 forward 边界只需保存 `chunk[:,-1,:,:head_dim]`（Ca），Cb 不再被读 ——
这就是 `overlap_kv` / `overlap_gate` 存在的理由。

对照官方实现：官方用 `overlap_transform`（`flash_model.py:307`）+ `coff = 1 + overlap`
（`:292`）表达同一件事，但把它塞在一个 tensor 的两半里；Transformers 把它拆成
两个显式持久化的 `overlap_*` 状态。

### 3.4 缓存层的自注册机制

缓存类通过类属性 `_layer_type` 注册到框架，`config` 由
`get_layer_types_and_kwargs`（`cache_utils.py:1753-1754`）注入
—— 这解释了为何它们的 `__init__` 签名是 `(config, **kwargs)` 而不是常规缓存签名。

### 3.5 Transformers 侧的 V4 能力落点

| 能力 | 符号 | 位置 |
|---|---|---|
| HC 展开 | `hidden_states.unsqueeze(2).expand(-1,-1,hc_mult,-1)` | modular:1225 |
| HC 混合 | `DeepseekV4HyperConnection` | modular:781 |
| HC 收尾 | `DeepseekV4HyperHead` | modular:895 |
| 交错 RoPE | `DeepseekV4RotaryEmbedding`（**无** `cat([freqs,freqs])` 复制） | modular:79 |
| 部分旋转 | `partial_rotary_factor = 64/512` | config:147 |
| 双 RoPE 分型 | `rope_parameters = {"main","compress"}` + `layer_type` 参数 | modular:116, config:294-321 |
| 分组输出投影 | `DeepseekV4GroupedLinear`（block-diagonal bmm） | modular:278 |
| HCA 压缩器 | `DeepseekV4HCACompressor` | modular:310 |
| CSA 压缩器 | `DeepseekV4CSACompressor` | modular:537 |
| 索引器打分 | `DeepseekV4IndexerScorer`（`weights_proj` + all_reduce） | modular:394 |
| 索引器 | `DeepseekV4Indexer` | modular:410 |
| Hash 路由 | `DeepseekV4HashRouter`（`tid2eid` buffer, `persistent=True`） | modular:977 |
| TopK 路由 | `DeepseekV4TopKRouter`（`e_score_correction_bias`） | modular:960 |
| EP 切分计划 | `base_model_ep_plan`（gate→`ep_router`，experts→`grouped_gemm`，indexer→`colwise`/`all_reduce`） | config:114-137 |
| 仅 eager | `_supports_flash_attn/_sdpa/_flex_attn = False`（:1095-1097）、`_can_compile_fullgraph = False`（:1105）、`_is_stateful = True`（:1136） | modular:1095-1136 |

### 3.6 值得注意的实现取舍（Transformers 版）

- **`n_shared_experts` / `topk_method` / `norm_topk_prob` / `attention_bias` /
  `router_jitter_noise` / `num_key_value_heads` 在代码中未被读取**（grep 确认）：
  `noaux_tc` 语义被 `e_score_correction_bias`（`:965,971`）硬编码替代，
  共享专家固定 1 个，`sqrtsoftplus` 作为普通逐元素激活直接套用（`:970,997`）。
- **`_can_record_outputs` 只登记了 `TopKRouter`，未登记 `HashRouter`**（待确认是否影响输出）。
- **公共 kv/bias 语义**：`s_aux=self.sinks`（每头可学习注意力汇，`:701,763`）传给注意力接口，
  在 eager 路径下是否被真正消费需按 backend 确认。

---

## 4. vLLM 适配调用结构

代码根目录：[repos/vllm/](../repos/vllm/)，commit `c91dccc08044b1269f351ceec13070fe8f075e10`
（vLLM 把 V4 实现放在 **非传统路径** `vllm/models/deepseek_v4/`，而不是 `vllm/model_executor/models/`）

### 4.1 注册与平台分派链（从模型名字到实现类）

```
HF config.architectures = ["DeepseekV4ForCausalLM"]
  │
  ├─ vllm/model_executor/models/registry.py:95
  │    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")
  │
  ├─ vllm/models/deepseek_v4/__init__.py:10   from vllm.platforms import current_platform
  │     ├─ :17  if current_platform.is_rocm()  →  .amd.model / .amd.mtp / .amd.dspark
  │     ├─ :26  elif current_platform.is_xpu() →  .xpu.model / .xpu.mtp / .xpu.dspark + .vl_stub
  │     ├─ :33  elif current_platform.is_cpu() →  .cpu.model / .cpu.mtp / .cpu.dspark + .vl_stub
  │     └─ :40  else (CUDA)                    →  .nvidia.model / .nvidia.mtp / .nvidia.dspark
  │                                              + .common.vl_model
  └─ __all__ (:50-56): DeepseekV4ForCausalLM / DeepSeekV4MTP /
                       DSparkDeepseekV4ForCausalLM /
                       DeepseekV4ForConditionalGeneration / DeepseekV4FP8Config
```

registry 中 V4 相关的**全部**入口（`vllm/model_executor/models/registry.py`）：

| 注册键 | 目标模块:类 | 行号 |
|---|---|---|
| `DeepseekV4ForCausalLM` | `vllm.models.deepseek_v4:DeepseekV4ForCausalLM` | :95 |
| `DeepseekV4ForConditionalGeneration`（多模态） | `vllm.models.deepseek_v4:DeepseekV4ForConditionalGeneration` | :365 |
| `DSparkDraftModel`（投机解码 draft） | `vllm.models.deepseek_v4:DSparkDeepseekV4ForCausalLM` | :637 |
| `DeepSeekV4MTPModel`（MTP） | `vllm.models.deepseek_v4:DeepSeekV4MTP` | :670 |
| `DeepseekV41ForCausalLM`（V4.1，另一实现） | `vllm.models.deepseek_v41:DeepseekV41ForCausalLM` | :369 |

> V4 与 V4.1 在 vLLM 中是**两套并行实现**（`deepseek_v4/` 与 `deepseek_v41/`），
> V4.1 额外有 `common/engram.py`、`decoder_replay_layers.py`、`mhc.py` 等。

### 4.2 `DeepseekV4ForCausalLM` 调用图

```
DeepseekV4ForCausalLM.forward(...)                 nvidia/model.py:2005
├─ self.model(input_ids, positions, ...)           :1417  DeepseekV4Model.forward :1551
│  ├─ get_pp_group().is_first_rank → embed_input_ids / inputs_embeds   :1558-1565
│  ├─ [sequence parallel] sp_shard(hidden_states / input_ids)          :1577-1578
│  ├─ [mega MoE] prepare_mega_gate_routing_metadata(...)               :1582 → :107
│  ├─ for idx, layer in islice(self.layers, start_layer, end_layer):   :1594
│  │   └─ DeepseekV4DecoderLayer.forward(...)                          :1598 → :1320
│  │      ├─ [首层] mhc_pre_broadcast_tilelang(...)  或 mhc_pre_tilelang(...)   :1336 / :1352
│  │      ├─ [后续层] mhc_fused_post_pre_tilelang(...)   ← post+pre 融合      :1366
│  │      ├─ sp_all_gather → attn → sp_reduce_scatter                  :1386-1390
│  │      ├─ DeepseekV4Attention.forward(positions, x, None)           :1388 → attention.py:471
│  │      │   ├─ _alloc_attn_out(...)                                  :479 → :503
│  │      │   ├─ _run_parallel_input_projections(hidden_states)        :485 → :672
│  │      │   │   └─ execute_in_parallel(
│  │      │   │        _fused_wqa_wkv_gemm,                    ← 主 stream
│  │      │   │        [compressor_kv_score,                   ← aux stream 0
│  │      │   │         indexer_weights_proj,                  ← aux stream 1
│  │      │   │         indexer_compressor_kv_score],          ← aux stream 2
│  │      │   │        ln_events, aux_streams, enable=...)             :720-731
│  │      │   ├─ _split_qkv_and_norm(qr_kv)                            :487 → :519
│  │      │   ├─ _prepare_and_attn_fn(...)   ← 捕获图内              :489
│  │      │   │   ├─ _prepare_and_attn / _prepare_and_attn_eager       :568 / :539
│  │      │   │   ├─ _sparse_indexer_and_attn(...)  ← @eager_break_during_capture  :734
│  │      │   │   │   └─ DeepseekV4Indexer.forward(...)                :928 → :1069
│  │      │   │   │        · _wq_b_proj                                  :1140
│  │      │   │   │        · DeepseekCompressor.forward(...)            compressor.py:354
│  │      │   │   └─ forward_mqa(q, kv, positions, output)  ← 平台实现  attention.py:170
│  │      │   ├─ attn_out[:, :n_local_heads, :]                         :493
│  │      │   └─ _o_proj(attn_out, positions)   ← 逆 RoPE + wo_a + wo_b :499 → :181
│  │      ├─ mhc_fused_post_pre_tilelang(... hc_ffn_fn ...)             :1394
│  │      └─ self.ffn(x, input_ids, mega_gate_metadata)                 :1413
│  │           └─ DeepseekV4MoE.forward(...)                           :1038
│  │               ├─ self.gate(hidden_states) → router_logits          :1062
│  │               ├─ fused_topk_bias(..., scoring_func, hash_indices_table=self.gate.tid2eid,
│  │               │                   routed_scaling_factor, ...)      :1064-1082
│  │               └─ [非 mega] _forward_fused_moe(...)                 :1052
│  ├─ idx+1 in aux_hidden_state_layers → mhc_post_tilelang → aux_hidden_states  :1607-1616
│  ├─ mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)    :1622
│  ├─ [PP 非末层] return IntermediateTensors                            :1626
│  ├─ [SP] sp_all_gather                                          :1634-1635
│  ├─ [MTP] _mtp_hidden_buffer.copy_(hidden_states.flatten(1))          :1637-1639
│  ├─ hc_head_fused_kernel_tilelang(...)   ← HC 头部收缩                :1641
│  └─ self.norm(hidden_states)                                          :1650
├─ self.lm_head(...) / compute_logits(...)                              :1992
└─ load_weights(...)                                                    :2023
   ├─ expert_mapping = self.get_expert_mapping()                        :2033
   └─ stacked_params_mapping（权重名融合表）  见 §4.4                     :1656+
```

### 4.3 vLLM 侧的 V4 特有机制落点

| 机制 | vLLM 符号 | 位置 |
|---|---|---|
| Hyper-Connections | `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang` / `mhc_post_tilelang` / `hc_head_fused_kernel_tilelang` / `mhc_pre_broadcast_tilelang` | `nvidia/model.py:1336,1352,1366,1394,1609,1622,1641` |
| HC 参数 | `hc_attn_fn` / `hc_ffn_fn` / `hc_attn_base` / `hc_ffn_base` / `hc_attn_scale` / `hc_ffn_scale` / `hc_eps` / `hc_sinkhorn_iters` / `hc_post_alpha` | `nvidia/model.py:1230-1268`, `:1426-1440` |
| Compressor | `DeepseekCompressor` + `CompressorStateCache` + `CompressorMetadataBuilder` + `CompressorBackend` | `compressor.py:200,139,92,60` |
| 压缩比驱动 | `compress_ratios[layer_id]`（`self.compress_ratio`） | `attention.py:232-235` |
| 压缩比归一化 | `self.compress_ratio = max(1, config.compress_ratios[layer_id]) if layer_id < num_hidden_layers else 1` | `attention.py:232-235` |
| 滑窗大小 | `self.window_size = config.sliding_window`（**全仓唯一读取点**） | `attention.py:221` |
| 融合压缩 GEMM | `compressor.fused_wkv_wgate`（把 `wkv`+`wgate` 合成一次 GEMM） | `attention.py:698`, `model.py:1668` |
| Indexer | `DeepseekV4Indexer` + `DeepseekV4IndexerCache` | `attention.py:928,882` |
| Indexer 权重投影 | `indexer.weights_proj`（`ReplicatedLinear`） | `attention.py:711` |
| Indexer topk 缓冲 | `topk_indices_buffer`（所有 Indexer 层复用） | `nvidia/model.py:1461` |
| 稀疏 MLA 后端 | `DeepseekV4SparseMLABackend` / `DeepseekV4FlashMLABackend` / `DeepseekV4FlashInferMLASparseBackend` | `sparse_mla.py:42,292`；`nvidia/flashinfer_sparse.py:104` |
| 后端选择 | `_select_dsv4_attn_cls(vllm_config)`：SM12 → FlashInfer，否则 FlashMLA | `nvidia/model.py:1146` |
| 滑窗 topk（短上下文） | `_fill_short_context_topk_indices` | `attention.py:75` |
| FP8/ue8m0 量化 | `DeepseekV4FP8Config`、`_resolve_dsv4_kv_cache_dtype`、`use_fp8_ds_mla_layout` | `quant_config.py`；`attention.py:93,191` |
| MTP | `DeepSeekV4MTP`、`get_mtp_target_hidden_states`、`_mtp_hidden_buffer` | `nvidia/mtp.py`；`model.py:2017,1637` |
| 投机 draft（DSpark） | `DSparkDeepseekV4ForCausalLM` | `nvidia/dspark.py` |
| 多模态 | `DeepseekV4ForConditionalGeneration`、`common/vl_model.py`、`common/vision.py` | `__init__.py:41` |
| NVIDIA 专用内核 | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`、`nvidia/ops/*cutedsl.py`、`_fused_qnorm_rope_kv_insert` | `attention.py:759` |

**vLLM 的三大工程化改造**（相对官方参考实现）：

1. **多流并行输入投影**：`_run_parallel_input_projections`（`attention.py:672`）
   把 4 个输入 GEMM 拆到 1 主 + 3 辅助 CUDA stream，
   并在 `x.shape[0] <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 时才启用。
2. **融合权重布局**：`fused_wqa_wkv`（`wq_a`+`wkv`）、`fused_wkv_wgate`（`wkv`+`wgate`）、
   `gate_up_proj`（`w1`+`w3`）——见 `packed_modules_mapping`（`model.py:1937-1941`）
   与 `load_weights` 的 `stacked_params_mapping`（`model.py:1656-1663`）。
3. **HC 与 norm 的融合内核**：官方是 `hc_pre` Python 实现（`flash_model.py:673`），
   vLLM 换成 `mhc_*_tilelang` / `mhc_fused_post_pre_tilelang`，
   把 `hc_post`+下一层 `hc_pre`+`RMSNorm` 融成一个 kernel（`model.py:1366`）。

### 4.4 KV cache / state 布局（三类 cache）

vLLM 把 V4 的 KV 状态拆成**三类独立 cache**（而非官方的一根拼接连 buffer）：

| cache | 类（定义处） | KV 规格 | 关键点 |
|---|---|---|---|
| 主 MLA / 压缩 KV | `DeepseekV4Attention`（`attention.py:131`） | `MLAAttentionSpec(..., head_size=head_dim, tokens_per_state=compress_ratio, alignment=576/512, model_version="deepseek_v4", state_content_bytes=584 if fp8_ds_mla)` | `compress_ratio <= 1` → 返回 `None`（`:857-860`）；584 B = 448 NoPE + 128 RoPE + 8 scale（`:876-878`） |
| Indexer K cache | `DeepseekV4IndexerCache`（`attention.py:882`） | `MLAAttentionSpec(..., head_size=head_dim, dtype=uint8, tokens_per_state=compress_ratio)` | `head_dim` 已含 FP8 scale padding（`:908-909`） |
| Compressor state | `CompressorStateCache`（`compressor.py:139`） | `SlidingWindowMLASpec(..., dtype=float32, sliding_window=coff*compress_ratio)`；`state_dim = 2*coff*head_dim`（`:283`） | 前一半存 `kv_state`，后一半存 `score_state` |

写入路径（与官方 §2.3(2) 的 `kv_state`/`score_state` 语义一一对应）：

| 写入对象 | 方法 | 位置 |
|---|---|---|
| compressor state（kv 半 + score 半） | `_SAVE_PARTIAL_STATES_KERNEL(kv, score, ape, positions, state_cache, slot_mapping, block_size, state_width, compress_ratio, ...)` | 调用 `compressor.py:399-410`；kernel `common/ops/save_partial_states.py:19`（`kernel:37`） |
| compressed KV / indexer K（compress→RMSNorm→RoPE→FP8/MXFP4→写） | `compress_norm_rope_store_fn(...)` | 调用 `compressor.py:475-499`；选择逻辑 `:447-473` |
| SWA cache（Q norm+RoPE；KV RoPE+量化+插入） | `_fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)` | 定义 `attention.py:759-847`，调用 `:593` |
| decode 期 topk 索引全局化 | `compute_global_topk_indices_and_lens(...)` | `common/ops/cache_utils.py:547-575` |

**一处必须纠正的常见误解**：`update` / `store_compression_weights` 这两个方法名
**只存在于 Transformers 实现**（`modular_deepseek_v4.py:167,181`），
在 vLLM 仓库中 grep 无任何匹配。vLLM 的等价职责由上述 kernel 化方法承担。

### 4.5 权重名映射（`load_weights`，`nvidia/model.py:1656`）

```python
stacked_params_mapping = [
    ("gate_up_proj",            "w1",     0),
    ("gate_up_proj",            "w3",     1),
    ("attn.fused_wqa_wkv",      "attn.wq_a", 0),
    ("attn.fused_wqa_wkv",      "attn.wkv",  1),
    ("compressor.fused_wkv_wgate", "compressor.wkv",   0),
    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
]
```

另有 `_make_deepseek_v4_weights_mapper(expert_dtype)`（`:1837`）按 `expert_dtype`
（`fp4` / `fp8`）切换 HF→vLLM 的权重名映射；`Pro-Base` 的 `expert_dtype = "fp8"` 即走这条分支。

### 4.6 配置字段 → vLLM 符号 对照

| `config.json` 字段 | vLLM 消费点 |
|---|---|
| `hidden_size` | `DeepseekV4Model.hidden_size`；`hc_dim = hc_mult * hidden_size`（:1444） |
| `num_hidden_layers` | `make_layers(config.num_hidden_layers, ...)`（:1466） |
| `hc_mult` | `self.hc_mult`（:1444），`mhc_*` 内核参数 |
| `hc_eps` / `hc_sinkhorn_iters` | `mhc_pre_tilelang(..., self.hc_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters, ...)`（:1352-1361） |
| `index_topk` | `topk_indices_buffer = torch.empty(max_num_batched_tokens, config.index_topk)`（:1461） |
| `index_n_heads` / `index_head_dim` | `DeepseekV4Indexer.__init__`（attention.py:929） |
| `compress_ratios` | `DeepseekV4Attention.compress_ratio`（`attention.py:232-235`）：`cr==4 ⇒ 建 Indexer；cr>1 ⇒ 建 Compressor；cr<=1（含被 max(1,·) 抬升的 0）⇒ 纯 SWA`；`get_kv_cache_spec` 对 `cr<=1` 返回 `None`（`:856-860`） |
| `sliding_window` | `attention.py:221`（`self.window_size`，全仓唯一读取点）+ `_fill_short_context_topk_indices`（`:75`） |
| `num_hash_layers`（HF 键名） | `Gate.tid2eid`；`DeepseekV4Model.has_local_hash_moe`（`nvidia/model.py:866,1218,1476`）；`DeepseekV4Model.forward:1584` |
| `scoring_func="sqrtsoftplus"` | `fused_topk_bias(..., scoring_func=self.scoring_func, ...)`（:1067） |
| `num_experts_per_tok` | `self.n_activated_experts`（:1075） |
| `routed_scaling_factor` | `fused_topk_bias(..., routed_scaling_factor=...)`（:1081） |
| `expert_dtype` | `_make_deepseek_v4_weights_mapper(expert_dtype)`（:1953）；`DeepseekV4MegaMoEExperts` |
| `o_groups` / `o_lora_rank` | `_o_proj` 的分组低秩输出投影（attention.py:181） |
| `q_lora_rank` | `_fused_wqa_wkv_gemm`（attention.py:647） |
| `head_dim` / `qk_rope_head_dim` | `_split_qkv_and_norm` / `_fused_qnorm_rope_kv_insert`（attention.py:519,759） |
| `vision_n_layers` | `__init__.py` 的 VL 分支；`mm_prefix_clamp_sliding_window`（`transformers_utils/configs/deepseek_v4.py:47`） |

### 4.7 vLLM 对 Transformers config 的扩展

`vllm/transformers_utils/configs/deepseek_v4.py:8` 定义 `DeepseekV4Config(PreTrainedConfig)`，
`model_type = "deepseek_v4"`，除 HF 字段外**额外注入两个滑窗修正量**（`:47-51`）：

```python
self.mm_prefix_clamp_sliding_window   = vision_n_layers > 0
self.mm_prefix_span_leading_pad_modulus = 4 if vision_n_layers > 0 else 0
```

即**多模态前缀（图片 span）会让滑窗范围变宽**，需要 kernel 层配合。

---

## 5. SGLang 适配调用结构

代码根目录：[repos/sglang/](../repos/sglang/)，commit `bb9a820f09f97480bfc6d07564fb2b691b8857a3`

### 5.1 注册机制：`EntryClass` 自动发现（与 vLLM 的显式 dict 不同）

SGLang 不在注册表里硬编码类名，而是**扫描模块内的 `EntryClass` 变量**：

```
python/sglang/srt/models/registry.py:29
    self.models = import_model_classes("sglang.srt.models") | self.models
        └─ :117-150  pkgutil 扫描 sglang.srt.models 包下每个模块
        └─ :153-159  用 cls.__name__ 作为注册键

python/sglang/srt/models/deepseek_v4.py:5899       EntryClass = [DeepseekV4ForCausalLM]
python/sglang/srt/models/deepseek_v4_nextn.py:261  EntryClass = [DeepseekV4ForCausalLMNextN]
python/sglang/srt/models/deepseek_v4_dspark.py:1264 EntryClass = [DeepseekV4ForCausalLMDSpark]
```

**配置层的架构改写**（`python/sglang/srt/configs/model_config.py`）：

```
is_deepseek_v4(config)                                        :275
  return _hf_arch(config) in ("DeepseekV4ForCausalLM",
                              "DeepseekV4ForCausalLMNextN",
                              "DeepseekV4ForCausalLMDSpark")
      ├─ 引用点：:277-279                                              (架构白名单)
      └─ :869-889  is_draft_model 时改写 architectures[0]:
             checkpoint_bundles_dspark_draft(...) and speculative_algorithm in (None,"DSPARK")
               → "DeepseekV4ForCausalLMDSpark"                       :882
             else
               → "DeepseekV4ForCausalLMNextN"                        :888
                 (并设 num_nextn_predict_layers = 1)                  :889
```

**参数钩子**（`python/sglang/srt/arg_groups/`）：

```
arg_groups/model_overrides/deepseek_v4.py:15  @_register_for("DeepseekV4ForCausalLM")
  _deepseek_v4_overrides(server_args, hf_config)
      └─ overrides = {"attention_backend": "dsv4"}              :22
           【注意】这是 V4 的默认注意力后端名 "dsv4"
arg_groups/deepseek_v4_hook.py:23  apply_deepseek_v4_defaults(server_args, model_arch)
      ├─ run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)   :43
      └─ validate_deepseek_v4_cp(server_args)                   :109
```

### 5.2 `DeepseekV4ForCausalLM` 调用图

```
DeepseekV4ForCausalLM.forward(...)                          models/deepseek_v4.py:5186
├─ DeepseekV4Model.forward(input_ids, positions, forward_batch, input_embeds, ...)   :4784
│  ├─ embed_tokens(input_ids) → hidden_states.unsqueeze(1).repeat(1, hc_mult, 1)      :4792-4796
│  │        ↑ 与官方 flash_model.py:805 的 HC 展开等价（形状 [tokens, hc, d]）
│  ├─ _check_late_layer_tail_readers(forward_batch)                                   :4444
│  ├─ _forward_layers_hc_pre_from_prev(...)                                           :4466
│  │   │   或 _forward_layers_tbo(...)   ← TBO = Two-Batch Overlap                    :4694
│  │   └─ DeepseekV4DecoderLayer.forward(...)                                         :3107
│  │      ├─ [融合路径] apply_mhc_post_pre_boundary(...)                              :3133
│  │      │     分支级联：aiter HIP(gfx95) → Triton → TileLang → None
│  │      ├─ [非融合] hc_post(...) → hc_pre(...)                                      :3162,3166 → :3017,2857
│  │      ├─ _input_norm(...) → x_quant                                               :3156 → :2838
│  │      ├─ MQALayer.forward(x, positions, forward_batch, x_quant)                   :2143
│  │      │   ├─ 多流派发判定 enable_multi_stream / low_ratio_multi_stream             :2160-2195
│  │      │   ├─ _forward_prepare(...)                                                :1785
│  │      │   │   或 _forward_prepare_multi_stream(...)                               :1444
│  │      │   │      / _forward_prepare_low_ratio_multi_stream(...)                   :1508
│  │      │   │      / _forward_prepare_multi_stream_npu(...)                         :1572
│  │      │   │      / _forward_prepare_multi_stream_hip(...)                         :1672
│  │      │   │   ├─ _normalize_q_lora / _compute_q_a / _compute_q_b                   :1282,1321,1332
│  │      │   │   ├─ _compute_kv_to_cache / _compute_kv_bf16                          :1379,1423
│  │      │   │   └─ Compressor（self.compressor）                                    :(MqaAttentionBase)
│  │      │   ├─ op_attn(state)   ← 注意力后端调用                                     :2644
│  │      │   │   └─ attn_backend = get_attn_backend()   ("dsv4")                     :2153
│  │      │   │        └─ DeepseekV4AttnBackend.forward(q,k,v,layer,forward_batch)    :3271
│  │      │   │             ├─ init_forward_metadata*                                 :2317 起
│  │      │   │             │   ├─ init_forward_metadata_decode                        :1317
│  │      │   │             │   ├─ init_forward_metadata_prefill                       :1334
│  │      │   │             │   ├─ init_forward_metadata_indexer                       :1272
│  │      │   │             │   ├─ init_forward_metadata_target_verify                 :1715
│  │      │   │             │   ├─ init_forward_metadata_dspark_draft_block            :1766
│  │      │   │             │   ├─ init_forward_metadata_in_graph / out_graph         :2022,2122
│  │      │   │             │   └─ init_forward_metadata_for_breakable_cuda_graph_capture :2543
│  │      │   │             └─ forward_low_ratio_sources(...)                         :2692
│  │      │   └─ _project_wo_b(o)   ← 分组低秩输出投影                                 :2593
│  │      ├─ op_gather_a / op_gather_b                                                :4168,4206
│  │      ├─ op_mhc_post_attn_pre_mlp(state)                                          :4057
│  │      ├─ op_moe(state)                                                            :4212
│  │      │   └─ _run_moe_ffn_dp_sync(...)                                            :3839
│  │      └─ op_mhc_postprocess(state)                                                :4134
│  ├─ hc_head(...)   ← HC 头部收缩                                                    :4408
│  └─ [MTP] DeepseekV4ForCausalLMNextN.forward                                        :deepseek_v4_nextn.py:211
└─ load_weights(weights, is_nextn=False)                                              :5482
   ├─ remap_weight_name_to_dpsk_hf_format(...)                                        :5344
   ├─ post_load_weights(is_nextn, weight_names)                                       :5308
   ├─ precompile_kernels_after_loading()                                              :5329
   └─ _prewarm_mhc_kernels()                                                          :5414
```

### 5.3 SGLang 的显式 `op_*` 流水线（重要设计差异）

SGLang 把 decoder layer 的 forward 拆成一组**显式命名的阶段函数**，
以便在 CUDA Graph 里按 stage 重排、跨层重叠：

| 阶段 | 方法 | 行号 | 职责 |
|---|---|---|---|
| 1 | `op_mhc_prepare_attn` | :4016 | HC 混合 + 注意力输入准备 |
| 2 | `op_gather_a` | :4168 | 通信/收集 A 段 |
| 3 | `op_attn` | `MQALayer:2644` | 注意力（含 compressor / indexer） |
| 4 | `op_gather_b` | :4206 | 通信/收集 B 段 |
| 5 | `op_mhc_post_attn_pre_mlp` | :4057 | attn 后 HC post + MLP 前 HC pre |
| 6 | `op_moe` | :4212 | MoE FFN（含 DP 同步 `_run_moe_ffn_dp_sync`） |
| 7 | `op_mhc_postprocess` | :4134 | HC 收尾 |

对应两种整层驱动模式：
- `_forward_layers_hc_pre_from_prev`（:4466）—— HC 输入来自上一层
- `_forward_layers_tbo`（:4694）—— **TBO（Two-Batch Overlap）**，把两个 batch 的
  注意力与 MoE 交错执行以隐藏通信

### 5.4 SGLang 的显存池与压缩状态

```
mem_cache/deepseek_v4_memory_pool.py
├─ DeepSeekV4SingleKVPool(KVCache)          :193
├─ DeepSeekV4UniformFP8KVPool               :339   ← 统一 FP8 KV
├─ HiSparseC4DevicePool                     :393
├─ DeepSeekV4IndexerPool(KVCache)           :501   ← Indexer 独立 KV 池
├─ DeepSeekV4UnifiedKVPool                  :781   ← 统一池（unified kv 路径）
└─ DeepSeekV4TokenToKVPool(BaseSWAKVPool)   :887   ← 滑窗(SWA) 语义的 token→KV 映射

mem_cache/deepseek_v4_compress_state.py
├─ KVAndScore (dataclass)                   :23    ← kv 与 score 成对状态
│   kv()/score()/from_kv_score()/cat()/clear() ...  :27-72
└─ CompressStatePool                        :129
    ├─ transfer_indices(req_pool_idx, seq_len)            :192
    ├─ _alloc_kv_score_buffer(...)                        :208
    ├─ state_cache_3d()                                   :241
    ├─ translate_from_swa_loc_to_state_loc(...)           :260
    └─ translate_from_req_position_to_state_loc(...)      :269
```

对比官方实现的 `Compressor.kv_state` / `score_state`（`flash_model.py:303-304`）：
SGLang 把它提升为**独立的池化状态 `CompressStatePool`**，
并引入 `translate_from_swa_loc_to_state_loc` 在"滑窗池坐标"与"压缩状态坐标"之间换算——
这正是 §2.3(2) 中"滑窗段与压缩段拼在一根 cache 里"的工程化结果。

### 5.5 SGLang 侧 V4 特有能力落点

| 能力 | 符号 | 位置 |
|---|---|---|
| 默认注意力后端 | `attention_backend = "dsv4"` | `model_overrides/deepseek_v4.py:22` |
| 注意力后端实现 | `DeepseekV4AttnBackend` / `DeepseekV4MultiStepBackend` | `deepseek_v4_backend.py:1062,4012` |
| HIP/ROCm 变体 | `DeepseekV4HipRadixBackend` | `deepseek_v4_backend_hip_radix.py` |
| TRT-LLM 变体 | `deepseek_v4_trtllm_backend.py` | 同目录 |
| MHC 内核调度 | `MhcOps` / `_get_mhc_ops()` / `apply_mhc_post_pre_boundary` | `models/deepseek_v4.py:210,220`；`amd/deepseek_v4_fused_mhc.py` |
| HC 头 | `make_hc_mixing_params` / `make_hc_head_params` / `hc_head_torch` | :578,595,606 |
| 低压缩比专用路径 | `low_ratio_multi_stream` / `deepseek_v4_low_ratio_sources` / `forward_low_ratio_sources` | :740；backend:2692 |
| Engram（哈希嵌入） | `deepseek_v4_engram_hash_ids` | :762 |
| wo_a FP8 GEMM | `wo_a_fp8_gemm_enabled` / `_apply_wo_a_bf16_matmul` / `_apply_gguf_grouped_wo_a` | :269,456,646 |
| MTP / NextN | `DeepseekV4ForCausalLMNextN` | `deepseek_v4_nextn.py:211` |
| DSpark 投机 | `DeepseekV4ForCausalLMDSpark` | `deepseek_v4_dspark.py:779` |
| 多模态 | `deepseek_v41_vit.py`、`multimodal/deepseek_v41_image_processing.py`、`multimodal/dsv41/vl_routing.py` | 同仓库（**注意：不存在 `multimodal/processors/deepseek_v41.py`**） |
| 工具调用/解析 | `deepseekv4_detector.py`、`rust/sglang-processor/.../deepseek_v4.rs` | 同仓库 |
| 内核 | `python/sglang/kernels/jit/csrc/deepseek_v4/*.cuh`（topk / compress / mhc / fp4 / fp8） | 32 个 `.cuh` |

### 5.6 配置字段 → SGLang 符号 对照

`python/sglang/srt/configs/deepseek_v4.py:48` 的 `DeepSeekV4Config(PretrainedConfig)` dataclass
比 HF 原版**多出以下字段**（V4 运行期需要，HF config.json 里没有）：

| 追加字段 | 默认 | 用途 |
|---|---|---|
| `kv_lora_rank` | 512 | MLA 的 KV 低秩（HF 用 `head_dim` 表达） |
| `qk_nope_head_dim` | 448 | 非 RoPE 头维 |
| `v_head_dim` | 512 | V 头维 |
| `first_k_dense_replace` | 0 | 前 N 层用 dense FFN |
| `moe_layer_freq` | 1 | MoE 层频率 |
| `n_group` / `topk_group` | 8 / 8 | `noaux_tc` 分组 |
| `ep_size` | 1 | 专家并行度 |
| `window_size` | 128 | 对应 HF `sliding_window` |
| `kv_source_layer_ids` / `index_source_layer_ids` | `[]` | **层间 KV/Index 复用**：某层直接借用别层的 KV |
| `candidate_source_layer_id` / `candidate_topk_blocks` / `candidate_block_size` | -1 / 0 / 0 | 候选块稀疏注意力 |
| `engram_layer_ids` / `engram_num_embeddings` / `engram_max_ngram_size` / `engram_vocab_size` / `engram_n_heads` / `engram_head_dim` / `engram_pad_token_id` | `[]`/… | Engram（n-gram 哈希嵌入） |
| `hc_pre_from_prev_sublayer` | False | HC pre 的输入取自上一层 |
| `q_head_norm` | True | Q 侧归一化（对应官方 `flash_model.py:498`） |
| `vision_*` / `image_token_id` | 0 / 129264 | 视觉塔 |

其中 **`kv_source_layer_ids` / `index_source_layer_ids` / `candidate_*` 是官方 config.json 中
完全不存在的字段**，属于 SGLang 为跨层复用压缩 KV 做的扩展。

### 5.7 SGLang 实现的关键事实校正（重要）

SGLang 的 V4 支持与 vLLM 差异很大，有几点**容易误判**，必须显式记录：

1. **`Configs/deepseek_v4.py:48` 的 `DeepSeekV4Config` 在本仓库中从未被实例化。**
   它只作为**类型注解**出现（`models/deepseek_v4.py:47`、`layers/attention/dsv4/indexer.py:32`、
   `layers/attention/dsv4/compressor.py:17`）。
   运行期 `hf_config` 的真实加载链在 `sglang/srt/utils/hf_transformers_utils.py`
   （**不在本次 sparse 检出内，未验证**），而运行期分支直接读 HF 原始属性
   （`configs/model_config.py:1229-1243`）。
   → 因此 §5.6 表格中的"字段"应理解为 **schema 声明**，不保证运行期由该类提供。

2. **`models/deepseek_v4.py` 只有 6 个顶层类**，压缩器/索引器/MoE 都不在其中：

   | 顶层类 | 行号 |
   |---|---|
   | `MhcOps`（NamedTuple） | `:210` |
   | `MqaAttentionBase` | `:771` |
   | `MQALayer` | `:1065` |
   | `DeepseekV4DecoderLayer` | `:2678` |
   | `DeepseekV4Model` | `:4273` |
   | `DeepseekV4ForCausalLM` | `:4947` |

   其余组件位置：
   - MoE → `models/deepseek_v2.py:2718`（**复用 V2 的 MoE**）
   - Compressor → `layers/attention/dsv4/compressor.py:331`
   - C4 Indexer → `layers/attention/dsv4/indexer.py:1111`
   - V4.1 Compressor / Indexer → `layers/attention/dsv4/dsv41_sparse.py:112 / :184`
   - Engram → `layers/engram.py:896`

3. **滑窗大小在 backend 里是硬编码的。**
   `SWA_WINDOW = 128`（`layers/attention/deepseek_v4_backend.py:138`），
   而 `config.window_size`（`configs/deepseek_v4.py:102`）**运行期不被读取**；
   实际读的是 HF 键 `sliding_window`：`configs/model_config.py:1237`
   `self.window_size = self.hf_config.sliding_window`。

4. **`head_dim` 被硬断言为 512**：`deepseek_v4_backend.py:1102-1104`。

5. **`n_hash_layers` 字段名与消费者不匹配（潜在缺陷）。**
   `DeepSeekV4Config.n_hash_layers` 定义在 `configs/deepseek_v4.py:133`，
   但 MoE 侧读的是 `num_hash_layers`：
   `models/deepseek_v2.py:634` `n_hash_layers = getattr(config, "num_hash_layers", 0)`。
   本次检出的范围内**未发现改名桥接点** → 只有 HF 侧确实提供 `num_hash_layers`
   （如 §1.1 所述 HF `config.json` 正是这个键名）时才能正确工作。
   vLLM 侧同样使用 `num_hash_layers`（`nvidia/model.py:866,1218,1476`），
   两框架一致，而 SGLang 自己的 dataclass 字段名是唯一的例外。

6. **`compress_ratios` 的取值域更大**：SGLang 支持 `{0, 1, 2, 4, 128}`，
   比官方/HF 的 `{0, 4, 128}` 多出 `1` 与 `2`：

   | ratio | 行为（SGLang） | 位置 |
   |---|---|---|
   | 4 | 建 `C4Indexer` + `Compressor` | `models/deepseek_v4.py:1145-1174` |
   | 128 | 只建 `Compressor` | 同上 |
   | 1 / 2 | **仅当该层在 `kv_source_layer_ids` / `index_source_layer_ids` 中**才建 `DeepseekV41Compressor` / `DeepseekV41Indexer` | `:1177-1191` |
   | 0 | 纯 SWA | — |

   且 `kv_source_layer_ids` 同时界定"late layer"SWA 有界 replay 的分界
   （`:4393-4396`，`late_layer_start = max(kv_source_layer_ids) + 1`）。

7. **`candidate_*` 三个字段不在模型文件中被消费**，
   消费者在 backend 与索引器子包内：
   `deepseek_v4_backend.py:1128-1134`、`dsv41_sparse.py:204-205`、
   `layers/attention/dsv4/v41_indexer/*`、`layers/attention/dsv4/low_ratio_backend_hip.py:133-192`。

---

## 6. 统一调用链对比

### 6.1 四份实现的同构骨架

把四份代码抽象后，DeepSeek-V4 的调用链是**同一张图**，差异只在每层的"实现载体"：

| 阶段 | 官方 `flash_model.py` | Transformers | vLLM | SGLang |
|---|---|---|---|---|
| 配置 | `ModelArgs`(:35) | `DeepseekV4Config` | `DeepseekV4Config`（含 VL 扩展） | `DeepSeekV4Config`（含 engram/复用扩展） |
| 注册 | — | `AutoConfig`(`model_type`) | registry dict(:95) | `EntryClass`(:5899) |
| 主体 | `Transformer`(:769) | `DeepseekV4Model` | `DeepseekV4Model`(:1417) | `DeepseekV4Model`(:4273) |
| 层 | `Block`(:647) | `DeepseekV4DecoderLayer` | `DeepseekV4DecoderLayer`(:1191) | `DeepseekV4DecoderLayer`(:2678) |
| HC 收缩 | `hc_pre`(:673) | 见 §3 | `mhc_pre_tilelang`(:1352) | `hc_pre`(:2857) + `apply_mhc_post_pre_boundary`(:3133) |
| HC 写回 | `hc_post`(:683) | 见 §3 | `mhc_fused_post_pre_tilelang`(:1366) | `hc_post`(:3017) |
| 注意力 | `Attention`(:436) | 见 §3 | `DeepseekV4Attention`(ABC, attention.py:131) | `MqaLayer`(:1065) |
| 压缩器 | `Compressor`(:279) | 见 §3 | `DeepseekCompressor`(compressor.py:200) | `Compressor`（`MqaAttentionBase` 内） |
| 索引器 | `Indexer`(:380) | 见 §3 | `DeepseekV4Indexer`(attention.py:928) | 见 §5.2 `low_ratio_sources` |
| MoE | `MoE`(:609) | 见 §3 | `DeepseekV4MoE`(:788) | `op_moe`(:4212)，路由/专家复用 `deepseek_v2.py:2718` |
| 路由 | `Gate`(:546) | 见 §3 | `fused_topk_bias`(:1064) | MoE 内路由 |
| 头部 | `ParallelHead`(:703) | 见 §3 | `DeepseekV4Model.hc_head_fused_kernel_tilelang`(:1641) + `lm_head` | `hc_head`(:4408) |
| MTP | `MTPBlock`(:738) | 见 §3 | `DeepSeekV4MTP`(nvidia/mtp.py) | `DeepseekV4ForCausalLMNextN` |
| 投机 | — | — | `DSparkDeepseekV4ForCausalLM` | `DeepseekV4ForCausalLMDSpark` |

### 6.2 三个"配置 → 行为"的关键跳变点

1. **`compress_ratios[layer_id]`**（官方 `flash_model.py:453`）
   → 决定该层是"纯滑窗 MLA"（0）、"重叠压缩 + 学习式索引器"（4）、
   还是"非重叠强压缩 + 固定 topk"（128）。
   vLLM 用 `DeepseekV4Attention.compress_ratio`、SGLang 用 `self.compress_ratio` 承接。

2. **`layer_id < n_hash_layers`**（官方 `flash_model.py:556`）
   → 前 3 层用 `tid2eid[input_ids]` 查表路由，其余层用 `sqrtsoftplus` 打分 topk。
   vLLM 用 `gate.tid2eid` + `hash_indices_table`（`model.py:1064-1082`）承接。

3. **`compress_ratio != 0` 才启用 YaRN**（官方 `flash_model.py:475-482`）
   → 同一模型内混用两套 RoPE 配置（10000 无 YaRN / 160000 + YaRN factor 16）。

### 6.3 事实清单（可直接引用的结论）

- 官方 Flash 与 Pro 的 `inference/model.py`、`inference/kernel.py` **完全相同**（MD5 一致），
  差异 100% 在配置；`Pro-Base` 与 `Pro` 只差 `expert_dtype`（`fp8` vs `fp4`）。
- `len(compress_ratios) == num_hidden_layers + num_nextn_predict_layers`，**末项属于 MTP 层且被官方实现真实读取**
  （`MTPBlock(n_layers+0)` → `Block` → `Attention` → `compress_ratios[43]`）。
  **三方独立验证**：V4-Flash `43+1=44`、V4-Pro `61+1=62`、
  [V4.1-Flash](DeepSeek-V4.1-架构与代码分析.md) `40+3=43`；
  V4.1 官方代码注释亦明确 *"one entry per layer, MTP layers included"*（`inference_model.py:80-81`）。
- Flash：层 0/1 无压缩；层 2..42 在 ratio 4 与 128 间交替；层 43（MTP）无压缩。
- Pro：层 0/1 为 ratio 128；层 2..60 在 4 与 128 间交替；层 61（MTP）无压缩。
- HC 的混合维度 `mix_hc = (2 + hc_mult) * hc_mult`，`hc_mult=4` → 24（官方 `:663`）。
- `Block.hc_pre` 用 **Sinkhorn**，`ParallelHead.hc_head` 用 **sigmoid + hc_eps**，两者不同（`：679` vs `:733`）。
- 只有 `compress_ratio == 4` 的层实例化 `Indexer`；ratio 128 用固定下标（`：466-471`）。
- `compress_ratio == 4` → `overlap = True`；128 → `overlap = False`（`：290`）。
- vLLM 把 V4 放在 `vllm/models/deepseek_v4/`（非 `model_executor/`），
  并按 ROCm/XPU/CPU/CUDA 四路分派实现（`__init__.py:17-48`）。
- SGLang 用 `EntryClass` 自动注册（`registry.py:29`），并把 decoder 拆成 7 个 `op_*` 阶段。
- SGLang 的 `DeepSeekV4Config` 独有 `kv_source_layer_ids` / `index_source_layer_ids` /
  `candidate_*` / `engram_*` 字段，为官方配置所无。
- vLLM 的 `DeepseekV4Config` 独有 `mm_prefix_clamp_sliding_window` /
  `mm_prefix_span_leading_pad_modulus`，为多模态滑窗修正所加。

---

## 7. 附录

### 7.1 本地目录结构

```
~/Desktop/deepseek-v4-analysis/
├── configs/
│   ├── Flash_config.json / Pro_config.json / Pro-Base_config.json
│   ├── Flash_inference_config.json / Pro_inference_config.json
│   ├── Flash_generation_config.json
│   └── official_inference/        # 官方参考实现（无权重）
│       ├── flash_model.py  flash_kernel.py  flash_generate.py  flash_convert.py
│       └── pro_model.py    pro_kernel.py
├── repos/
│   ├── vllm/   (sparse: vllm/models/deepseek_v4, vllm/models/deepseek_v41,
│   │            vllm/transformers_utils/configs, vllm/model_executor, ...)
│   └── sglang/ (sparse: python/sglang/srt/{models,configs,layers/attention,
│                mem_cache,arg_groups,function_call,managers}, kernels/ops)
└── analysis/
    ├── DeepSeek-V4-代码调用结构分析.md   ← 本文档（V4 总报告）
    ├── DeepSeek-V4-张量数据流详解.md     # V4 逐阶段张量/形状/KV 布局
    ├── DeepSeek-V4.1-架构与代码分析.md   # V4.1 增量架构与适配
    ├── DeepSeek-V4-官方资料与训练后训练补充.md  # 官方资料核验（训练部分边界说明）
    ├── DeepSeek-vLLM与SGLang-Server设计哲学对比.md  # serving/多硬件适配对比
    ├── DeepSeek-V4-复盘与流程图.html     # 入口索引与流程图总览
    └── parts/                            # 各方深度分析原始报告（含更细的行号引用）
        ├── 01-official-reference.md      # 官方参考实现（827 行 model + kernel）
        ├── 02-vllm.md                    # vLLM 适配（813 行，含类清单/KV cache/不确定项）
        ├── 03-sglang.md                  # SGLang 适配（449 行，~315 处引用）
        ├── 04-transformers.md            # Transformers 适配（668 行，含 10 项未验证点）
        ├── 05-vllm-tensors.md            # vLLM 侧张量数据流
        ├── 06-sglang-tensors.md          # SGLang 侧张量数据流
        ├── 07-vllm-v41.md                # vLLM 的 V4.1 适配
        └── 08-sglang-v41.md              # SGLang 的 V4.1 适配
```

> `parts/` 下的报告是**逐符号、逐行号**的原始分析，
> 本文档是其汇总与交叉验证；需要更细的引用时直接查阅对应 part 文件。
> 若只想先建立整体印象，建议从 [复盘与流程图](DeepSeek-V4-复盘与流程图.html) 入手。

### 7.1.1 相关分析文档

- [DeepSeek-V4 张量数据流详解](DeepSeek-V4-张量数据流详解.md)：端到端张量形状、KV cache 布局与字节估算。
- [DeepSeek-V4.1-Flash 架构与代码分析](DeepSeek-V4.1-架构与代码分析.md)：V4 → V4.1 的增量项、压缩 KV 与索引器。
- [DeepSeek-V4 / V4.1 官方资料核验与训练后训练补充](DeepSeek-V4-官方资料与训练后训练补充.md)：官方材料交叉核验与使用边界。
- [vLLM 与 SGLang Server 设计哲学对比](DeepSeek-vLLM与SGLang-Server设计哲学对比.md)：serving 架构、调度/KV 缓存与多 GPU 平台适配。

### 7.2 复现命令

```bash
# 1) 官方配置与参考实现（不下载权重）
for f in config.json generation_config.json inference/config.json \
         inference/model.py inference/kernel.py inference/generate.py inference/convert.py; do
  curl -sL "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/$f" -o "$(echo $f | tr / _)"
done

# 2) vLLM / SGLang 稀疏克隆（见 §0.1）
```

### 7.3 未覆盖范围

- **权重文件**：按用户要求未下载（`*.safetensors`、`model.safetensors.index.json`）。
  因此无法做权重形状校验，配置字段与参数张量的对应关系仅依据代码推断。
- **vLLM 稀疏检出导致的结构盲区**：本次 sparse checkout 未包含 `vllm/v1/**`，
  因此以下符号**有使用点但无定义行号**，本文档未给出行号：
  `DeepseekV4SWACache`、`DeepseekSparseSWAMetadata/Builder/Backend`、`MLAAttentionSpec`、
  `SlidingWindowMLASpec`、`get_compressed_slot_mapping`、`dsa_indexer_uses_fp4`、
  `get_max_prefill_buffer_size`、`flash_mla_*`、
  `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`。
  同理 `vllm/models/common/ops/*`、`vllm/utils/multi_stream_utils.py` 亦不在检出内。
- **vLLM `deepseek_v41/`**：V4.1 为独立实现，本文仅记录其在 registry 中的入口（§4.1），
  未展开调用链。
- **SGLang `python/sglang/kernels/jit/csrc/deepseek_v4/*.cuh`（32 个 CUDA/HIP 内核）**：
  仅列出位置，未逐内核分析。
- **SGLang 未检出的内核包**：`sglang.kernels.ops.attention.dsv4.unified_kv_kernels`
  （被 `models/deepseek_v4.py:2196` 引用）不在本次 sparse 路径内。
- **`csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`**：
  仅记录其对应的三个 Python 入口（`attention.py:800/820/834`），未逐行分析 CUDA 实现。
- **`pro_model.py` / `pro_kernel.py`** 与 Flash 版字节完全相同，不重复分析。

