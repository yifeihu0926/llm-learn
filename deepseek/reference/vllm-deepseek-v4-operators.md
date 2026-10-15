# vLLM DeepSeek V4：实际算子总结

依据本地 vLLM **c91dccc08044b1269f351ceec13070fe8f075e10**，以 NVIDIA 文本主干为主，附可选路径与其他平台概览。相关源码在本次阅读时无本地修改。这里总结的是从forward追到的实际调用入口及其可达分支；没有运行GPU profiler，具体部署的一次执行子集仍由硬件、配置、shape与安装扩展决定。

## 1. 先把算子放回模型

![vLLM DeepSeek V4 主干与Attention算子路径](../assets/vllm-v4-operators-map.svg)

**读图。** 首层先把embedding作为重复C路残差的来源；mHC pre收缩并融合子层输入Norm。Attention输出经post/pre进入MoE；MoE结果携带residual/mix返回，交给下一层开头回写。最后一层独立post，然后hc_head收束、最终Norm，compute_logits再投词表。图中每个框可能包含多个kernel，也可能与相邻步骤部分融合。

记号：T是该调用实际token行数（可含padding，SP下可能是本rank shard）；D为hidden，C=hc_mult，I为expert intermediate，E为专家数，K_e为每token激活专家数，K_i为Indexer选择的历史压缩数。主Q/KV head_dim=512、RoPE尾部64；Flash/Pro使用同一实现，具体层数和宽度从config读取。

| 架构模式 | 本实现的ratio | 实际增加的工作 | 聚合入口 |
|---|---|---|---|
| SWA | 配置0映射到内部1 | 局部因果窗口索引 | 所选sparse MLA后端 |
| CSA / C4A | 4 | 主KV压缩、Indexer K压缩、Q评分、历史top-k | 同一后端，增加压缩候选 |
| HCA / C128A | 128 | 粗粒度KV压缩、枚举全部可见摘要 | 同一后端，增加粗摘要候选 |

CSA的Indexer是Attention内部步骤；MoE Router选择专家；Sampler选择词表token。这三个top-k处理不同对象。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/attention.py#L733)

## 2. 按模块检索实际入口

每行按一个入口或功能组整理，复合函数可以启动多个kernel。搜索与模块按钮会筛选本节的主干、条件分支及可选路径表；它们不是“同一次前向全部运行”的launch数量。

<!-- ops-toolbar -->

### 2.1 [通用] Embedding、动态量化与普通 Linear

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

### 2.2 [mHC] 四路残差、系数生成与融合

记 C=hc_mult，X 为 [T,C,D]；系数投影输出 C(C+2) 个值，拆成 pre、post 与 C×C residual mix。NVIDIA decoder 直接使用下面的 TileLang 封装。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 首层广播 pre | `mhc_pre_broadcast_tilelang` | [T,D] → 保存 C 路 residual，并生成已 RMSNorm 的 [T,D] 子层输入。 | 首个 PP rank 首次 2D 输入；预合并广播投影权重。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L698) |
| prenorm 投影 | `tf32_hc_prenorm_gemm` / `_HC_PRENORM_GEMM_TILELANG_KERNEL` | 计算 v·Fᵀ 和平方和；用于归一化系数投影。 | 普通 pre：DG 可用用 TF32 helper，否则 TileLang；广播 pre 强制 DG helper。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L22) |
| gate / Sinkhorn / 收缩 | `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` | inverse RMS → sigmoid pre/post gates → Sinkhorn → C路加权收缩 → layer-input RMSNorm。 | pre 及 post/pre 后续步骤；decoder 传入 Norm 权重以融合归一化。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L542) |
| post/pre 复合调用 | `mhc_fused_post_pre_tilelang` / `_MHC_FUSED_TILELANG_KERNEL` | 旧子层输出回写 residual，并准备下一子层的 pre；之后仍运行 big_fuse。 | 本 rank T≤32 且 D%128=0 用融合前段；否则拆成 post + prenorm GEMM。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L814) |
| 独立 post | `mhc_post_tilelang` / `_MHC_POST_TILELANG_KERNEL` | X_new[j] = post[j]·u + Σ_i mix[i,j]·X_old[i]，输出 [T,C,D]。 | 最后一个本地 decoder；aux hidden 重建；post/pre 的拆分分支。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L791) |

mHC 中 residual 为 BF16，系数/关键统计为 FP32，prenorm 可用 TF32 GEMM。FFN 的 post 留给下一层开头；本地循环末尾才独立重建。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1320)

### 2.3 [Attention] 投影、Q/KV 准备、索引与稀疏聚合

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

### 2.4 [Compressor] 历史状态与压缩 KV

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

### 2.5 [Indexer] CSA 的评分与历史 top-k

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

### 2.6 [MoE] 路由、专家执行与共享专家

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

路由专家的理想计算为 a=SiLU(xW_gateᵀ)⊙(xW_upᵀ)，贡献为 w·aW_downᵀ。原生MegaMoE把w放在L2前，并在加权后量化中间激活；最终combine不再乘w。详细流程见[源码导读](mega-moe-source-guide.html)。

### 2.7 [输出] 主干收束、LM Head 与采样

主模型forward产hidden；compute_logits和Sampler属于之后的独立调用。runner通常只选择需要出token的hidden行投词表。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 多流head收缩 | `hc_head_fused_kernel_tilelang` | [T,C,D]→learned sigmoid权重→[T,D]，FP32关键统计。 | 最后PP rank；完成最后mHC post之后。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/kernels/mhc/tilelang.py#L1041) |
| 最终Norm | `RMSNorm` / `ir.ops.rms_norm` | 最后一维RMS归一化；保持[T,D]。 | hc_head之后；batch-invariant时选专用实现。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/layernorm.py#L74) |
| 词表投影 | `ParallelLMHead` / `LogitsProcessor._apply_head` | [T_selected,D]→词表logits；TP普通路径gather并去padding。 | generation默认model dtype；显式FP32 head_dtype时可torch.mm(out_dtype=FP32)。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/logits_processor.py#L136) |
| 采样与logprobs | `Sampler` / `TopKTopPSampler` / `argmax` | logits转FP32；按配置做penalty、temperature、top-k/top-p与随机或greedy采样。 | serving runner调用；speculation则进入验证/rejection路径。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/v1/sample/sampler.py#L73) |

compute_logits_local(skip_gather=True)返回本rank带padding的词表shard，不是完整全词表。采样前cast FP32也不意味着LM Head矩阵乘法默认FP32。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1999)

### 2.8 [通信] SP / TP / EP 的数据动作

通信可在模型层、runner或融合后端内部发生。按output_is_reduced等条件决定归约，不能把所有collective无条件叠加。

| 算子组 | 实际入口 | 数据与计算作用 | 启用条件 | 定位 |
|---|---|---|---|---|
| 本地token分片 | `sp_shard` / `sp_padding_mask` | padding + token维本地slice；本身不通信。 | PP=1、EP开启、TP>1，且mega家族或DP>1时启用本模型SP。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/common/ops/sequence_parallel.py#L42) |
| SP gather | `sp_all_gather` | Attention前收集token shard；模型末尾收集完整C路hidden。 | 本模型SP启用；优先custom collective，否则TP all_gather。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/common/ops/sequence_parallel.py#L23) |
| SP求和并分片 | `sp_reduce_scatter` | Attention TP partial结果求和并按token分片。 | SP启用；对应wo_b.reduce_results=False，避免再独立all_reduce。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/common/ops/sequence_parallel.py#L30) |
| 普通MoE通信/归约 | `dispatch_router_logits` / `combine` / `tensor_model_parallel_all_reduce` | 按EP runner或internal MoE kernel完成token交换及输出归约。 | 受DP/SP、supports_internal_mk、output_is_reduced等条件控制；原生/FI mega内部执行专用通信。 | [源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L774) |

PP边界交付已重建的[T,C,D] IntermediateTensors，send/recv由执行器安排，model.py没有直接launch这些网络操作。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/nvidia/model.py#L1626)

### 2.9 [可选] MTP、DSpark 与图片输入

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



## 3. Prefill 与 Decode 的具体变化

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

## 4. 后端和精度怎样决定算子

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

## 5. 其他平台与模型边界

| 平台入口 | 当前V4主要实现 | 适用边界 |
|---|---|---|
| NVIDIA | TileLang mHC、CUDA/Triton/CuTe融合、FlashMLA/FI、所选MoE | 上述详细清单的主体 |
| ROCm | AMD模型、AITER sparse MLA、AITER/其他MoE与mHC分支 | 独立硬件实现；不能套NVIDIA DeepGEMM/NVLink执行细节 |
| XPU | XPU Q/KV insert、FP8 decode dequant+sparse attention、xpu_fp8_bmm、_xpu_C mHC | VL入口当前为NotImplemented stub |
| CPU | CPU fused cache/compressor/indexer/MLA、BF16 packed O BMM、CPU mHC | Indexer为FP8-only；VL入口当前为NotImplemented stub |

平台入口按is_rocm→is_xpu→is_cpu→默认NVIDIA分支导出。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/__init__.py#L17)

本篇针对**V4**。V4也有DSpark；classic MTP、DSpark是两种独立draft路径。V4.1自己的Full/Reindex/Reuse、Engram、candidate selector、hc_collapse及视觉packed graph接入不应并入普通V4清单。当前V4视觉wrapper逐图eager调用共享ViT/Aligner；共享文件中存在的packed/DP helper不等于该wrapper已调用它。[源码](https://github.com/vllm-project/vllm/blob/c91dccc08044b1269f351ceec13070fe8f075e10/vllm/models/deepseek_v4/common/vl_model.py#L200)

## 6. 源码定位与执行范围

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

[返回总文档](../archive/deepseek-v4-framework.html#section-46) · [MegaMoE源码导读](mega-moe-source-guide.html) · [V4量化详解](../archive/deepseek-v4-quantization.html)
