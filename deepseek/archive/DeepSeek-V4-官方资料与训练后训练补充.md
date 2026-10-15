# DeepSeek-V4 / V4.1 官方资料核验与训练/后训练补充

> **2026-10-07 本机复核更新**：核心结构、参数/缓存口径与 CPU 边界用例见 [本机核验与更正](DeepSeek-V4-V4.1-%E6%9C%AC%E6%9C%BA%E6%A0%B8%E9%AA%8C%E4%B8%8E%E6%9B%B4%E6%AD%A3.md)。旧文中的 `[未检出]` 是写作时的检出状态；当前可用路径以新报告为准。GPU 性能、训练与完整模型质量未在本机复现。

> 本文作为本地架构/代码分析的资料层补充，重点区分「官方公开声明」「本地源码可验证事实」与「公开资料未披露」。本文不把 V4.1 的训练细节外推为 V4 的训练配方。
>
> 本地代码/配置检查日期：以现有笔记记录为准（V4 代码抓取约 2026-10-02；V4.1 README 已保存于 `configs/v41/README.md`）。网络资料通过搜索索引取得；直接抓取若受 DNS/源站访问限制，本文明确标出，建议复核链接的当前版本。

## 2026-10-07 补充状态

V4 arXiv 正文与 V4.1 51 页技术报告现已成功取得。旧表中的“未取得/未阅读”仅反映当时状态；当前已核对 V4 §4/§5、V4.1 §2.2/§3.2.2/§4.2.2/§5.1/§5.2。新增训练细节与证据定位统一见 [本机核验与更正](DeepSeek-V4-V4.1-%E6%9C%AC%E6%9C%BA%E6%A0%B8%E9%AA%8C%E4%B8%8E%E6%9B%B4%E6%AD%A3.md)。V4 公共训练摘要 PDF 仍未取得，不臆测其正文。

## 1. 官方资料索引与权威性

| 资料 | 链接 | 适合回答的问题 | 本地核验状态 |
|---|---|---|---|
| V4 技术报告 | [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence（arXiv HTML）](https://arxiv.org/html/2606.19348)；[官方 Hugging Face 仓库](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 架构、训练方法、评测方法、消融与技术定义 | 搜索索引确认有此报告；本环境未能直接抓取正文，以下不声称逐段复核全文 |
| V4 Flash 模型卡 | [deepseek-ai/DeepSeek-V4-Flash README](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 型号定位、推理说明、配置/权重入口 | 搜索索引确认 README；本地有 HF config 和参考推理实现副本 |
| V4 Pro 模型卡 | [deepseek-ai/DeepSeek-V4-Pro README](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | Pro 版本信息和技术报告链接 | 搜索索引确认模型仓库；本地保存 Pro config 与推理参考实现 |
| V4 公共训练内容摘要 | [Public Summary of Training Content for DeepSeek-V4（PDF）](https://cdn.deepseek.com/policies/DeepSeek_Template_for_the_Public_Summary_of_Training_Content_for_GeneralPurpose_AI_model.pdf) | 依法公开的训练内容类别/摘要 | 搜索索引确认该官方 PDF 存在；PDF 正文未能在本环境抓取，**不摘录未核对的细节** |
| V4 API 发布说明 | [DeepSeek V4 Preview Release](https://api-docs.deepseek.com/news/news260424/) | API 发布时的产品能力与服务口径 | 搜索索引可见；API 资料属于产品/服务信息，不等同于训练报告 |
| V4.1 官方模型卡 / README | [DeepSeek-V4.1-Flash README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md) | V4.1 架构、预训练/后训练概要、评测设置、编码协议 | 本地保存版本已直接检查 `configs/v41/README.md` §Introduction、Evaluation、Prompt Encoding |
| V4.1 技术报告 | [DeepSeek_V41_Tech_Report.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | 详细方法、训练和评测 | 本地架构笔记记载 PDF resolve 下载异常、未阅读报告；本补充仍将其列为待核验 |
| V4.1 官方发布说明 | [DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient（API Docs）](https://api-docs.deepseek.com/news/news260910/) | 发布摘要与产品能力 | 搜索索引确认；具体技术细节优先以模型卡和报告为准 |
| V4.1 官方代码仓库 | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 配置、官方推理/编码实现、评测脚本 | 本地 `configs/v41/` 包含 README、配置、inference、encoding 相关文件（未必是完整仓库镜像） |

### 证据标记

- **[OFFICIAL-README]**：官方模型卡/README 明确写出。可引用，但 README 摘要不一定等于完整训练报告。
- **[LOCAL-CODE]**：在本地保存的官方参考代码或框架代码里直接观察到。只对保存的版本成立。
- **[DERIVED]**：由参数、公式、源码路径推导；需保留推导前提。
- **[UNVERIFIED]**：有官方资料入口，但内容尚未成功读取或没有足够证据。

## 2. V4 与 V4.1：官方公开材料交叉核验

| 主题 | V4 官方公开材料 / 本地证据 | V4.1 官方 README 的公开声明 | 复盘结论 |
|---|---|---|---|
| 模型定位 | V4 报告题目聚焦高效百万 token 上下文；本地 V4 config/参考实现覆盖 Flash、Pro | 多模态 MoE，最高 1M context；CED 架构 | 不能把 V4.1 的 CED/图像/训练规模视作 V4 特征 |
| 主干规模 | 本地配置与形状估算：Flash ≈284B，Pro ≈1.6T；属于配置/形状核算，报告原文仍应复核 | V4.1 Flash README 声称 552B backbone，另列 Engram 196B | 明确「backbone」和 Engram 条件记忆是否计入总参数口径 |
| 激活参数 | 本地 V4 文档基于模型结构和配置核算 Flash ≈13B、Pro ≈49B；应回查报告术语口径 | README 声称 prefill 8B / decode 16B | activated 参数不是总参数；V4.1 prefill/decode 激活口径也不能互换 |
| 上下文与稀疏注意力 | V4 本地官方 config/实现展示 SWA、压缩 KV、Indexer 等路径；报告中长度与训练阶段细节待全文核验 | 64K 长度训练稀疏注意力；在看到 34T tokens 时扩展到 1M context | README 是明确公开摘要；对 V4 不可推断同样的 45T/34T 配方 |
| 预训练规模 | 现有本地 V4 摘要没有充分记录可核验的 token 总量/阶段配方 | README 明确：从头训练，多模态语料 45T tokens；在 34T tokens 时扩展到 1M | 45T、34T 是 V4.1 官方 README 声明；训练顺序/配方细节需技术报告支持 |
| 后训练 | 现有 V4 本地代码分析主要是推理实现，不包含后训练算法证据 | README 概括 SFT → RL → OPD，称未改算法，创新重点在数据管线 | V4.1 只知范式级摘要；不能据此虚构 RL 算法、奖励模型、数据配比或训练超参 |
| Agent 数据 | V4 具体任务合成/rollout 规模在现有材料中未确认 | V4.1 README 称大规模自动合成 agent 任务与环境，并渐进扩展数据、任务、rollouts | 「渐进式扩展」是 README 声明；具体数量与筛选标准未见于目前核验材料 |
| 推理努力度 | V4 具体接口行为须查 V4 API/模型卡版本 | V4.1 支持整数 1–100 连续 reasoning effort；README instruct 结果使用 effort=100 | 模型推理参数不等于后训练算法；evaluation 表格注明 temperature=1、top_p=.95 |
| Prompt 协议 | V4 本地代码调用报告记载官方 encoding 路径曾未完整收集，需看版本 | V4.1 README 说明该发布不含 Jinja chat template，提供 encoding Python reference 与 deepseek-recipe | V4.1 tool/thinking/image prompt 不能简单用普通 chat template 代替 |

### 需要澄清的数字口径

1. **V4.1 552B + 196B Engram**：README 将 552B 称 backbone，并将 Engram 作为独立组件列出。不能在不说明口径时直接说模型「总计 552B」；按本地结构推导，把 Engram 纳入总参数后约 749B，但这是 **[DERIVED]**，须注明 Engram 稀疏访问及是否纳入特定官方统计口径。
2. **45T tokens 与 34T tokens**：V4.1 README 的语义是「以 45T token 多模态语料从头训练；在训练到 34T tokens 时把 context 扩至 1M」。这不是「额外再训练 34T」，也不代表 1M 全程训练。
3. **8B/16B 激活量**：V4.1 README 分别标为 prefill/decode；不要与总骨干参数、活跃专家个数混为一谈。建议在笔记注明 FLOP/参数激活统计是否计入视觉、Engram、共享层和草稿头。
4. **KV bytes/token**：V4.1 README 的 890 B/token 是特定设计/口径；本地估算应逐项核对量化 scale、Indexer K、压缩层数、滑窗是否由 replay 重建、对齐和运行时池化开销。不能用逻辑 payload 代替真实 GPU allocation。

## 3. 训练 / 后训练：目前能负责任地总结到什么程度

### 3.1 V4.1 公开的高层训练流程

```text
多模态预训练（从头训练，README：45T tokens）
  ├─ DeepSeek-ViT 从头训练（2D-RoPE、3×3 pixel-unshuffle）
  ├─ 视觉投影器把图像变为视觉 embeddings，与文本 embeddings 联合进入 LM 预训练
  ├─ 稀疏注意力训练长度 64K
  └─ 到 34T tokens 时扩展上下文到 1M
        ↓
SFT
        ↓
RL
        ↓
OPD（on-policy distillation）
        ↓
指令 / Agent 能力与 reasoning-effort 控制
```

**证据边界：**上述顺序和总述来自 V4.1 官方 README 的 [Pre-training] / [Post-training] 段落（本地 `configs/v41/README.md:53-57`）。README 说后训练沿用标准 SFT → RL → OPD 范式、没有算法改动，并强调数据管线变化；没有在已读材料中披露每阶段的数据量、训练步数、优化器、学习率、奖励模型结构、RL 算法名、OPD teacher/student 具体配置。故不能补写成确定事实。

### 3.2 官方公开的后训练要点（V4.1）

- **SFT → RL → OPD**：这是模型卡公开的阶段顺序摘要，并非完整 recipe。
- **数据管线是主要变化点**：自动合成 Agent 任务和环境；渐进扩展数据规模、任务复杂度与 rollout。当前已读 README 没给具体曲线、样本数、合成模型、验证器或去重率。
- **reasoning effort 1–100**：面向推理时成本/能力的可控设置。README 称更高 effort 增加推理成本并可能提高准确性；instruct benchmark 表使用 `reasoning_effort=100`、temperature 1.0、top_p 0.95。需分清这是推理设置/训练后呈现出的能力，不是「训练算法为 1–100」。
- **Agent 评测 scaffold 会影响分数**：README 对 Terminal-Bench、DeepSWE、视觉 Agent 等指定不同 harness、context window、采样数及 steps。跨模型比较须对齐 harness，不能只比表格数字。
- **官方 prompt encoding 是协议组件**：README 指出没有 Jinja template，另提供 encoding reference / deepseek-recipe 支持 thinking、tool、图片及 reasoning effort。复现实测时应把协议版本、parser 和模型权重一并锁定。

### 3.3 V4（非 V4.1）的训练信息：待补而非类推

当前本地 V4 总结聚焦架构和 inference reference，不能仅从 MoE、压缩注意力或推理代码推断训练方案。需要从 V4 技术报告、官方模型卡和 Public Summary PDF 逐项补证：

- 预训练数据的来源类别、语言/代码/数学/多模态占比、过滤/去重/质量控制。
- 预训练 token 数、数据混合与课程安排、长上下文扩展阶段和 sequence length schedule。
- 训练基础设施、并行策略、优化器、批大小、学习率、稳定性技巧与算力；若官方未披露，标注未公开。
- Base → Instruct 的 SFT/RL/蒸馏阶段、数据类型、奖励信号、拒绝采样、安全训练和推理能力培养。
- Flash 与 Pro 的训练关系：是否独立训练、蒸馏/合并/不同数据或只是配置规模不同。不能因官方参考推理代码相同就断言训练 recipe 完全相同。
- 训练内容合规摘要：Public Summary PDF 的实际条目和适用对象。PDF 未读前不概括其内容。

## 4. 评测材料的使用边界

V4.1 README 提供 Base 与 Instruct 两张不同口径的表，并注明部分 benchmark 设置：shots、metric、reasoning effort、temperature、top_p，以及不同 agent scaffold 和 rollout N。整理 benchmark 时建议加如下记录字段：

| 字段 | 必须记录的内容 |
|---|---|
| 模型身份 | Base / Instruct、具体版本、权重 revision |
| Sampling | temperature、top_p、effort、seed、max tokens |
| Prompt | 官方 encoding / chat template revision、system prompt、工具描述 |
| Harness | agent scaffold、工具版本、容器与网络权限、最大步骤 |
| 采样统计 | 每题 N、pass@k / resolved 定义、置信区间/方差（若有） |
| 数据泄漏风险 | benchmark 时间、训练数据污染检查（如果官方披露） |
| 可比性 | 同榜各模型是否相同 harness、上下文限制和预算 |

README 中出现的自家 benchmark 分数是**官方报告结果**，不是独立第三方复现。特别是 agent benchmark，harness、工具预算、网络访问与上下文长度都会影响结果；需和独立复现分栏展示。

## 5. 建议下一步核验清单

### P0：先补源文件

1. 从官方 Hugging Face V4 Pro 仓库取得当前技术报告 PDF 和对应 README；保存 revision/commit、下载时间、SHA256，并抽取训练章节原文。
2. 成功取得 DeepSeek V4 Public Summary PDF，逐条记录内容，不从文件标题或二手文章推测。
3. 取得 V4.1 技术报告 PDF（本地旧记录称 resolve 异常），与现存 README 对照：确认训练/后训练数据声明有没有更完整说明。
4. 把官方材料版本写入一个 manifest：URL、revision、日期、hash、本地路径、读取状态。

### P1：训练章节结构化摘录

把训练资料分别整理为：模型/数据规模、数据治理、多模态融合、长上下文训练阶段、优化和并行、SFT、RL、蒸馏、Agent 数据合成、安全与评测。每段保持短引文/页码或小节号，避免纯二手复述。

### P2：独立复现与比较

对本地可运行模型记录端到端 prompt 协议、effort 级别、TTFT/TPOT、输出质量、cache 字节和不同 harness 的成功率；官方结果与本地测量分开，不使用本地测量反推训练 recipe。

## 6. 对现有笔记的具体更新建议

- 在 [V4 代码调用结构分析](DeepSeek-V4-%E4%BB%A3%E7%A0%81%E8%B0%83%E7%94%A8%E7%BB%93%E6%9E%84%E5%88%86%E6%9E%90.md) §0 增加「官方训练资料及未覆盖范围」链接。
- 在 [V4 张量数据流详解](DeepSeek-V4-%E5%BC%A0%E9%87%8F%E6%95%B0%E6%8D%AE%E6%B5%81%E8%AF%A6%E8%A7%A3.md) §7 的显存估算旁加注：这是 inference cache / weight shape 估算，不是训练显存或训练配方。
- 在 [V4.1 架构与代码分析](DeepSeek-V4.1-%E6%9E%B6%E6%9E%84%E4%B8%8E%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90.md) §0 / §6 更新 V4.1 技术报告当前下载状态；保留「尚未阅读」标识直到全文逐节核验完成。
- 在复盘 HTML 中新增官方资料/训练后训练导航入口；醒目标注 V4 vs V4.1 的训练声明不可互相外推。

## 7. 来源链接（可直接打开）

- [V4 技术报告（arXiv）](https://arxiv.org/html/2606.19348)
- [V4 Flash 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [V4 Pro 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [V4 公共训练内容摘要 PDF](https://cdn.deepseek.com/policies/DeepSeek_Template_for_the_Public_Summary_of_Training_Content_for_GeneralPurpose_AI_model.pdf)
- [V4 API 预览发布说明](https://api-docs.deepseek.com/news/news260424/)
- [V4.1 Flash README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md)
- [V4.1 Flash 技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
- [V4.1 Flash 官方发布说明](https://api-docs.deepseek.com/news/news260910/)
