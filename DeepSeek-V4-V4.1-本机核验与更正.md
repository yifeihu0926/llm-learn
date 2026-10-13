# DeepSeek V4 / V4.1 本机核验与更正

核验日期：2026-10-07（Asia/Shanghai）。对象是本机保存的源码和资料快照。本文更正旧笔记的核心结构、数字和证据范围，提供可复跑记录；没有加载完整权重或运行完整模型质量/性能评测。

## 1. 结论与重要更正

原笔记对 V4 的三类 attention、V4.1 的跨层 KV / index 共享、Engram 与 DSpark 的大方向基本正确；以下几项需要实质性修正。

| 项目 | 旧结论的问题 | 本次核验结果 |
|---|---|---|
| V4.1 模式层数 | Reuse=29，总数仅39 | **SWA=2、Full=4、Reindex=4、Reuse=30**，合计40 |
| V4 MTP 配置 | parts 中称最后一个 ratio 未读，与主文冲突 | MTP 构造器确实读取最后项；Flash 43+1=44、Pro 61+1=62、V4.1 40+3=43 |
| `tid2eid` 转换风险 | 白名单笔误被推导成真实参数丢失 | `tid2eid` 输入名在本次名称变换中完整保留；笔误存在，但不足以证明其未初始化 |
| V4 主干输出 logits | 一些流程图写 `[b,s,V]` | 参考普通生成只取最后位置，**`[b,V]`**；DSpark full logits 另有全位置输出 |
| V4.1 DSpark 参数 | 0.54B / 约1.1GB | 三个完整 stage 额外 **14.225362530B**，构造器存储 **8.099641736GB**（含scale，embed/head不重复） |
| V4.1 ViT 参数 | 0.32B | ViT **0.411842560B**，另有 aligner **0.073410560B** |
| V4.1 总存储 | 约489GB | 参考构造器主干等组件 **505.049GB**，含DSpark **513.149GB**；非实测部署内存 |
| 890 B/token | 容易被当作参考实现实际内存 | **packed global payload** 公式成立；参考实现 BF16 fake-quant cache 为 **3200 B/token**，另有SWA与state |
| prefill 8B / decode 16B | 用全主干活跃专家8.49B证明prefill8B | 不能这么证明。生产decoder bounded replay限制后半层prefill token数；参考代码整段跑40层 |
| bounded replay | 只说重建，容易理解为数值等价 | 技术报告明确是**近似重建**；encoder新suffix状态也随cache命中位置变化 |
| 分层候选索引 | “候选池使索引成本不依赖上下文”混用于参考实现 | 参考代码先对全历史dense打分再mask；生产稀疏算子才可能获得该省算 |
| 目标层隐藏态 | 称DSpark抓取attention输入 | 实际抓取**进入Block前的residual stream均值**，没有经过hc_pre或attn_norm |
| V4.1 index score | 表格称FP32，candidate维写S/2 | 默认BF16输入时score为BF16；层20 ratio=1，candidate维是S |
| “未检出” | 旧parts沿用早期稀疏检出范围 | 当前SGLang speculative、model_executor、quantization已存在；vLLM部分仍缺失，逐项见§7 |

## 2. 本地资料范围、版本与方法

发现的专题目录：`[本地源码路径]`。初始共有 **14份分析文档、10,581行**（13 Markdown、1 HTML），以及配置、参考实现、图片预览、vLLM / SGLang检出。另纳入 `[本地源码路径]` 中已安装的 Transformers V4 代码。Documents、Desktop其他目录与本地笔记路径的专题文本检索未找到额外独立分析文档。

这是一份已发现资料的审计，不声称扫描过操作系统所有缓存、应用数据库或未挂载磁盘。截图属于HTML的展示副本，不构成独立代码证据。

| 快照 | 核验版本 |
|---|---|
| vLLM | `c91dccc08044b1269f351ceec13070fe8f075e10` |
| SGLang | `bb9a820f09f97480bfc6d07564fb2b691b8857a3` |
| 原安装Transformers | 5.18.0；没有替换其环境 |
| 核验环境 | Apple M6、16 GiB；独立venv，Python 3.14、Torch 2.14.1、NumPy 2.5.3 |
| CUDA | 不可用；本轮数值用例全部在CPU运行 |

方法分为四种，结论严格限制在相应范围内：

1. **源码检查**：读取配置、构造器、状态更新与框架调用路径，确认条件分支和版本。
2. **原始构造器meta统计**：运行本地原始类构造器，分配meta张量，逐参数统计逻辑维度与存储字节。没有分配数百GB真实权重。
3. **CPU小用例**：执行原始纯PyTorch函数/方法，使用确定性的合成权重。V4 compressor测试跳过RoPE/量化，只验证池化与状态交接；V4.1 indexer测试将GPU fake-quant设为identity，验证dense路径、dtype和因果下标。
4. **新增官方资料核对**：取得V4报告HTML、V4.1报告PDF及官方encoding.py；保留来源快照。官方训练/质量/吞吐声明不变成本机复现结果。

未调用GPU kernel的测试以会抛错的stub隔离，只有明确声明的池化/indexer测试局部使用identity替代。没有把stub输出冒充kernel数值正确性。

## 3. 参数量与权重存储重新核算

统计规则：B=10^9逻辑参数，GB=10^9字节；FP4两个逻辑值压在一个元素中，计数时乘2。量化scale独立计数，计入存储但不混入逻辑模型参数；整数hash路由表另列。共享embed/head只计一次。

| 模型/口径 | 逻辑参数B | 参考构造器权重GB（含scale） |
|---|---:|---:|
| V4 Flash主干+embed/head，不含MTP | 284.332240471 | 159.117270492 |
| V4 Flash完整MTP额外部分 | 6.610048891 | 3.627384172 |
| V4 Pro主干+embed/head，不含MTP | 1572.997201763 | 858.243815884 |
| V4 Pro完整MTP额外部分 | 25.840145979 | 14.023768236 |
| V4.1 主干+Engram+视觉+embed/head，不含DSpark | 748.979953264 | 505.049432528 |
| V4.1 完整DSpark额外部分 | 14.225362530 | 8.099641736 |
| V4.1 全部独立组件 | 763.205315794 | 513.149074264 |

V4.1的Engram需要区分三个口径：

- lookup表：768,022,850行×256，**196.613849600B**；每行256B FP8+8B scale，**202.758032400GB**。
- 整个Engram模块：加上投影及gate参数，**196.928504320B**、**203.073076240GB**。
- 从748.979953264B中扣整个Engram，剩**552.051448944B**；只扣lookup表则剩552.366103664B。两者都在官方552B的近似量级内，但口径不能交换。

旧笔记里的“精确印证”应改为“量级一致”。参数总数近似吻合，不能证明DSpark、视觉塔或所有存储项正确。DSpark仍有完整MoE，不能只计main_proj、Markov和confidence小头。

构造器存储与checkpoint体积也不一样：参考实现将head提升为FP32、部分compressor提升为FP32、wo_a改为BF16；shared experts跟随默认FP8。上述数值不包含缓存、激活、workspace、allocator padding或通信buffer。也没有用它推断训练显存或实际TP部署显存。

## 4. KV/state与复杂度口径

### 4.1 890 B/token可以证实到什么程度

```text
main KV: 512/2 + 512/16 = 288 B/压缩位置
index K: 128/2 + 128/32 = 68 B/压缩位置
源层: ratio=2的层2/8/14 + ratio=1的层20
global payload: (288+68) × (3/2+1) = 890 B/原始token
```

本机执行SGLang `KVLayout.page_bytes`：V4 payload=584B，page_size=64/128时物理585B/token；V4.1 SWA=528B，FP4 main=288B，这两个布局在64/128页大小下不增加该行的padding。仍有请求尾页浪费、index池布局与元数据开销，**890不是完整allocator measurement**。

官方参考实现的cache仍是BF16数组，量化后就地回写BF16。其global cache为 `(512+128)×2×2.5=3200 B/token`；40层×128×512×2的SWA为 **5MiB/序列**。参考实现没有生产bounded replay的持久化逻辑，不能用其数组推导出实际890。

V4 Flash参考实现64K cache+compressor state合计 **468,729,856B=447.015625MiB/序列**，旧数值正确。但此合计含固定窗口与状态，不能整个当作持久global KV。

V4 Flash packed global基准随index-K格式变化：

```text
FP4 index-K: 584×(21/4+20/128)+68×21/4 = 3514.25 B/token
FP8 index-K: 584×(21/4+20/128)+132×21/4 = 3850.25 B/token
```

前者/890≈3.9486；后者/890≈4.3261。页对齐会进一步改变分配量。旧文把3850–3899直接断言为官方“四倍”的唯一口径，证据不足。

### 4.2 prefill / replay / 分层索引的区别

- 层20的source compressor使用经hc_pre / attn_norm处理的输入；后续层共享它生成的global KV。层19原始residual输出仍有`[b,s,hc,d]`维，不能直接写成已经收缩的`[b,s,d]`。
- 原参考`Transformer.forward`循环40层，对整个prefill执行。**CED共享本身不会自动把它变成8B prefill实现**。生产decoder replay只让后半部分处理尾部窗口，节省随N增长的计算；官方8B/16B是生产口径。
- 参考`Indexer.forward`先做全历史einsum，之后用candidate mask限制结果。候选池限制选取范围，不等于这份参考代码已经只对16,384个候选位置打分。需要生产sparse backend才能实现相应复杂度。
- Reuse省掉compressor/indexer，不等于整个attention“零成本”：该层仍执行Q投影、SWA、sparse attention和O投影。
- V4.1 bounded replay有两个独立机制：encoder replay帮助只缓存global KV；decoder replay省掉后半层大部分prefill。V4.1报告§3.2.2/p20明确二者都是近似重建，并在后训练适配；不能要求其与完整前向逐bit一致。

## 5. 本机运行验证

最终17组检查通过。详情和逐项证据见 [verification_results.json]([本地源码路径])。

| 验证 | 覆盖与结果 |
|---|---|
| 文件/版本审计 | 配置、参考实现、检出源码与原安装Transformers生成SHA256清单；核对HEAD；修正实际Markdown相对链接 |
| 层拓扑 | Flash/Pro/V4.1的主干+MTP长度；V4.1 Full/Reindex/Reuse/SWA计数 |
| Flash/Pro代码身份 | 本地model.py、kernel.py分别字节完全相同；只证明参考代码相同 |
| 原构造器参数 | 三种模型逐层meta构造；包括MoE、MTP、视觉、Engram及量化scale |
| 压缩器状态 | V4 ratio4/128、V4.1 ratio1/2；prefix取1、r−1、r、r+1；完整prefill与后续单token累积结果一致（池化范围） |
| SWA下标 | prefill长度1/3/4/5/127/128/129/257，窗口1/4/128；decode环回边界检查；无越界未来有效位置 |
| 候选block | 240组含零历史、部分block和最新block强制保留；partial block mask本身可覆盖未来位置，仍需causal mask |
| Engram hash | 合成identity tokenizer map下prefill/decode一致；图像DEAD barrier阻断跨图像ngram；不证明真实压缩tokenizer一致 |
| Head/MoE | 普通logits为[b,V]；非softmax路由权重和为route_scale；hash路由读固定表 |
| DSpark head | block5产生[b,6] ids（首项anchor+5项生成）、[b,5,V] logits、[b,5] confidence；未测试拒绝/接受循环 |
| 图像网格 | 微小、常见分辨率、极长宽比等7例；含start/end/newline总token数≤1024 |
| SGLang config | 执行原normalize函数，验证nested字段提升、vision命名和架构名映射到V4 |
| Indexer dense路径 | 原forward在identity fake-quant下运行；BF16 score、ratio1 candidate维与causal top-k |
| 编码对拍 | 4类prompt×chat/thinking×4档整数effort，共32例，local Jinja与官方Python字串一致 |
| 转换名称 | 原name/key规则验证真实tid2eid名保留；tie2eid名不会被自动更正；非完整权重转换测试 |
| Replay反例 | 两层窗口均值的独立数学用例：完整与bounded重算末项不同；不用于评价DeepSeek质量 |

**编码新增发现**：本地Jinja文件开头说明是从encoding.py移植，官方模型卡说没有官方Jinja。32例一致只能证明这些样例；它额外接受`medium=62`，本次下载官方encoding.py拒绝medium。不能把它直接称为官方全协议等价模板，tool多轮、generation特殊边界和token IDs还需额外对拍。

## 6. 补充训练与后训练资料

### V4

旧补充说报告正文未取得；现在已取得并核对§4/§5。报告声明预训练语料超过32T，包含数学、代码、网页、长文档等；后训练先独立训练领域专家，再用多teacher、全词表reverse-KL OPD整合，替换mixed RL整合阶段；后训练还对专家权重与indexer QK进行FP4 QAT。以上为作者报告，未在本机复现。[V4技术报告](https://arxiv.org/html/2606.19348)

### V4.1

已成功读取51页报告，保存PDF并检查相关页。旧笔记中的“报告下载异常、尚未阅读”不再是当前状态。

- **训练数据**：45T为总量，34T是上下文扩展节点。§4.1/p21还给出text-only与multimodal token混合 **7:1**。这与后训练大量合成Agent任务是不同阶段。
- **优化器与schedule**：§4.2.2/p22披露matrix用Muon、norm/非矩阵参数用AdamW、embedding/head用Sinkhorn-balanced更新；Engram学习率5倍。batch固定100.6M tokens；前2000步warmup，2.6e−4维持至28T，28T–40T cosine降至2.6e−5，40T–45T保持；34T扩展64K→1M。
- **视觉预训练**：§4.2.2/p22–23披露ViT独立contrastive预训练与autoregressive fine-tuning，再集成进语言骨干。“视觉塔从头训练”不等于仅随LM一步训练、完全没有前置阶段。
- **后训练**：§5.1/p25开始说明SFT→RL→OPD，主要变化在任务/环境生成、质量过滤、去重和难度校准。§5.2/p30–31披露sample级异步派发、training抢占rollout，以及跨checkpoint routing replay。
- **effort**：§5.1.4/p29–30给出effort条件与token penalty；有限训练档位支持部署插值。报告中的public API preset为low50/high75/max100，不能把本地模板medium别名视为官方API档位。
- **近似replay**：§3.2.2/p20是本次最重要的架构补充。报告说质量影响小属于作者实验；本机没有复现该质量结论。

来源：[V4.1技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)、[本地PDF快照]([本地源码路径])、[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md)。公开训练内容摘要PDF未取得，未补写其内容。

## 7. 逐份原资料处理与剩余边界

| 原资料 | 核验/处理 |
|---|---|
| V4代码调用结构 | 核对配置和构造链；Flash/Pro byte identity；修复17处相对链接；收窄“差异100%配置”措辞 |
| V4张量数据流 | cache447MiB成立；修正生成logits；替换参数/权重预算，补完整MTP与scale |
| V4.1架构与代码分析 | 修正模式数、参数/存储、capture语义、score dtype/candidate维、复杂度与格式限定；补PDF证据 |
| 官方资料与训练后训练补充 | 更新报告已取得状态；新训练细节见§6；保留历史未读取状态作为当时记录 |
| vLLM与SGLang Server设计哲学对比 | 固定HEAD和请求入口仍成立；“设计哲学”是解释性比较，不视为吞吐或功能优劣实测 |
| parts/01官方参考 | 更正末ratio未读矛盾；撤回tid2eid必然丢失推导 |
| parts/02 vLLM、03 SGLang | 主模型/后端组织与配置路径按快照检查；添加当前核验入口 |
| parts/04 Transformers | 原安装5.18.0与ratio0/4/128映射仍可直接查证；额外unknown kwargs等原观察项未冒充运行结果 |
| parts/05 vLLM tensors、06 SGLang tensors | cache布局/页对齐重新运行核对；历史未检出标记须结合当前可用路径 |
| parts/07 vLLM V4.1、08 SGLang V4.1 | 跨层共享结构/配置适配成立；补路径状态、DSpark消费者和生产/参考省算区别 |
| 复盘HTML与preview图片 | HTML加入核验入口；图片是旧HTML截图，不作独立实验证据，未重绘 |

当前路径实证：

| 路径 | 状态 |
|---|---|
| vLLM `v1/kv_cache_interface.py`、`distributed/` | 已存在 |
| vLLM `v1/spec_decode/`、`config/`、`utils/`、`models/common/` | 当前仍缺失 |
| SGLang `speculative/`、`model_executor/`、`layers/quantization/` | 已存在 |
| SGLang `layers/moe/`、`utils/hf_transformers_utils.py` | 当前仍缺失 |

SGLang `speculative/dspark_components/dspark_config.py:214-219`现已能直接确认block_size/noise_id消费者；因此旧文“SGLang全仓零引用”失效。未运行两个完整serving框架、GPU kernels、分布式collective、真实tokenizer压缩映射、权重加载、Agent benchmark或训练复现；这些不能标为本机已验证。源码行号是否语义正确也不因“处于文件范围内”就成立。

## 8. 交付与复跑

- [验证脚本]([本地源码路径])：原函数与构造器验证，GPU kernel默认禁止调用。
- [结果与证据]([本地源码路径])：17组检查、参数分类、cache、prompt对拍、版本与环境。
- [源码SHA256清单]([本地源码路径])：专题目录与原安装Transformers源文件；每次运行更新快照。
- [文档改动记录]([本地源码路径])：14份原文的before/after SHA256及本地备份位置。
- [链接审计]([本地源码路径])：修复后应为空；代码块伪链接不计。
- [依赖锁定]([本地源码路径])；[V4来源HTML]([本地源码路径])、[官方V4.1编码快照]([本地源码路径])。

本机复跑：

```bash
[本地源码路径] \
  [本地源码路径]
```

若迁移电脑，需要修改脚本ROOT与原Transformers路径，创建独立venv并安装requirements-lock；source路径和PyTorch版本必须随结果记录。不要为复跑下载完整权重。

原分析目录已同步本报告与更正，源码repo和原安装Transformers文件未修改；原笔记备份保存在本任务work/original_notes。补充材料的来源与SHA256另见 [external_source_manifest.json]([本地源码路径])。
