# vLLM 与 SGLang Server 设计哲学对比

> **2026-10-07 本机复核更新**：核心结构、参数/缓存口径与 CPU 边界用例见 [本机核验与更正](DeepSeek-V4-V4.1-本机核验与更正.md)。旧文中的 `[未检出]` 是写作时的检出状态；当前可用路径以新报告为准。GPU 性能、训练与完整模型质量未在本机复现。

> 对比对象是本地两个真实 Git checkout 的固定快照，不是泛指项目所有历史版本。
> - vLLM：`c91dccc08044b1269f351ceec13070fe8f075e10`
> - SGLang：`bb9a820f09f97480bfc6d07564fb2b691b8857a3`
> - 两个仓库均为 shallow/blobless + sparse checkout；server 相关路径已按上述 commit 加入 sparse checkout。代码不是只下载模型文件的临时副本。
>
> 行号以固定 checkout 的文件为准，更新仓库后可能漂移。本文只总结读到的主干路径，特殊配置/功能会改变路径。

## 1. 一句话总结

- **vLLM V1：以「通用、分层、可替换的执行引擎」组织 serving。** API/异步前端负责输入输出和 admission；EngineCore 集中调度 token progress、执行模型并管理 KV block 生命周期；Executor / EngineCoreClient 隔离硬件执行和进程通信。
- **SGLang SRT：以「面向请求与前缀状态的在线运行时」组织 serving。** HTTP/API 与 TokenizerManager 收口协议化请求和流式状态，通过进程间消息交给 Scheduler；Scheduler 将 continuous batching、prefill/decode 衔接、Radix prefix cache、KV 预算和 worker 执行放在紧密的运行时循环里，并提供 normal/overlap loop 等不同执行模式。

这不是「vLLM 没有 prefix cache」或「SGLang 没有分层」：vLLM 同样有 prefix caching、KV connectors、各种 executor；SGLang 同样有多个 manager、worker 和控制层。差异是**主要抽象边界和默认组织方式**，不能简化成特性有无。

## 2. 快照与代码范围

| 项目 | 固定 commit | 补拉/核查的 serving 路径 |
|---|---|---|
| vLLM | `c91dccc08044b1269f351ceec13070fe8f075e10` | `vllm/entrypoints/`、`vllm/v1/engine/`、`vllm/v1/core/`、`vllm/v1/executor/`、`vllm/v1/kv_cache_interface.py` 等 |
| SGLang | `bb9a820f09f97480bfc6d07564fb2b691b8857a3` | `python/sglang/srt/entrypoints/`、`managers/`、`mem_cache/`、`model_executor/`、`speculative/`、`observability/` 等 |

本次没有更改项目源代码，也没有运行模型或 benchmark。这里的「拉取」是在现有 Git checkout 上扩大 sparse-checkout 路径，HEAD commit 未改变。

## 3. 端到端请求路径图

```text
vLLM V1（典型 Async API / 多进程 EngineCore）

OpenAI Router / Serving
  → AsyncLLM.generate / add_request
  → InputProcessor + OutputProcessor + per-request collector
  → EngineCoreClient（inproc / sync MP / async MP 等 client 实现）
  → EngineCoreProc / EngineCore
  → Scheduler.schedule() → Executor.execute_model()
  → Scheduler.update_from_output()
  → EngineCore outputs → AsyncLLM output handler → collector → HTTP stream

SGLang SRT（典型 FastAPI + TokenizerManager + Scheduler）

FastAPI HTTP route
  → TokenizerManager.generate_request
  → normalize / validate / tokenize / request state
  → typed IO message over ZMQ / IPC
  → Scheduler.ingest_requests / Req / waiting queue
  → get_next_batch_to_run（prefix/cache/budget aware）
  → TP worker / model runner → GPU batch
  → scheduler process result / detokenizer & output message
  → TokenizerManager RID state / async waiter
  → SSE / HTTP response
```

**重要入口提醒：**此 vLLM commit 中 `vllm/entrypoints/openai/api_server.py` 只有兼容导出，已经 deprecated；它不是实际 route 实现。服务组装在 `vllm/entrypoints/launchers/api_server/entry.py` 与 routers/app 中。SGLang 的 `http_server.py` 是实际 FastAPI HTTP 入口之一。

## 4. vLLM V1：核心抽象与设计取舍

### 4.1 前端与 EngineCore 解耦

`AsyncLLM` 是异步引擎包装层，负责输入处理、请求提交、输出关联和客户端侧生命周期；定义可见 [`async_llm.py:80`](../repos/vllm/vllm/v1/engine/async_llm.py)。典型添加请求进入 EngineCoreClient（`async_llm.py` 中 `_add_request`，约 533–553 行），生成器读取 per-request collector 并流式产出（`generate`，约 666 行起）。后台 output handler 汇聚 EngineCore 输出，再交给 OutputProcessor/collector（约 793 行起）。

`EngineCoreClient` 抽象隐藏 EngineCore 的部署/通信模式；实际类型包括进程内、同步或异步多进程客户端。API 构造逻辑 `build_async_engine_client_from_engine_args` 创建配置及 AsyncLLM 并负责 shutdown，见 [`entry.py:69-114`](../repos/vllm/vllm/entrypoints/launchers/api_server/entry.py)。因此 HTTP serving 不必知道模型执行位于本进程还是独立 EngineCore。

### 4.2 EngineCore 是 token-progress inner loop

V1 Scheduler 明确不建模为两个僵硬阶段「prefill queue / decode queue」：每步尝试让 `num_computed_tokens` 追上 `num_tokens_with_spec`，以同一 token-progress 表示兼容 chunked prefill、prefix cache 和 speculative decode，见 [`scheduler.py:574-585`](../repos/vllm/vllm/v1/core/sched/scheduler.py)。调度先处理 running requests，在 token/input budget 下计算本轮应处理 token；KV slot 不能分配时可按策略 preempt running request（约 `scheduler.py:640-839`）。

EngineCore 执行一个 step 时，职责链是 scheduler 输出 → model executor → 输出采样/更新 scheduler（`core.py` 的 step，约 630–659 行）。可选 batch queue 允许异步提交多个批次，减轻 PP pipeline bubble，但输出消费和请求状态更新更复杂（[`core.py:670-783`](../repos/vllm/vllm/v1/engine/core.py)）。

### 4.3 KV cache 是有接口契约的 block 管理系统

`KVCacheManager` 对 Scheduler 隐藏内部 block 数据结构，以 group-oriented `KVCacheBlocks` 作为边界（[`kv_cache_manager.py:34-54`](../repos/vllm/vllm/v1/core/kv_cache_manager.py)）。它基于 KV specs 和 cache groups 做分配、释放、prefix hit 和外部 KV connector 协作。无法分配时返回无资源结果，由 scheduler 延后或抢占，而不是 allocator 自己决定请求策略。

BlockPool 以 hash/block/refcount 管理共享缓存块：prefix caching 对完整块做查找/复用，引用计数控制正在使用的数据与可驱逐数据。注意 prefix cache hit 通常不意味着可以跳过末尾 token 计算；实现还需保留最后位置 logits 等语义。外部 KV transfer、滑窗、分组 cache、Mamba 等规格会扩展行为。

### 4.4 哲学归纳

- 通过稳定组件边界管理复杂性：API / AsyncLLM / EngineCoreClient / EngineCore / Scheduler / Executor / Worker。
- 对调度状态使用通用 token-progress 与显式 `SchedulerOutput`，便于纳入多模态、编码器预算、draft token、KV transfer 等约束。
- 把 KV cache 抽象为 block/spec/group/coordinator，而不是让 scheduler 直接管理模型张量细节。
- 在一个 engine-core step 中集中做 admission、token budget、block allocation、preemption 和执行提交；可选择 async scheduling / batch queue 等模式。

## 5. SGLang SRT：核心抽象与设计取舍

### 5.1 HTTP/Tokenizer/Scheduler/Worker 进程职责分离

SGLang `http_server.py` 是 FastAPI 推理 HTTP 入口，文件说明在 [`http_server.py:14-18`](../repos/sglang/python/sglang/srt/entrypoints/http_server.py)。`/generate` 流式处理在 [`http_server.py:910-951`](../repos/sglang/python/sglang/srt/entrypoints/http_server.py)：route 调用 TokenizerManager 的 async generator，SSE 编码由 HTTP 层完成，断连交由 abort task/请求生命周期处理。OpenAI 等协议 route 将协议请求转换后仍走 runtime。

`TokenizerManager.generate_request` 负责规范化 batch/参数、创建 RID state、校验、处理 pause/weight-update 锁、tokenize，再发 tokenized request 并异步等回包，见 [`tokenizer_manager.py:859-931`](../repos/sglang/python/sglang/srt/managers/tokenizer_manager.py)。请求与控制消息用 typed IO structs 和 socket 通信；SRT 并非 HTTP handler 直接调用 GPU 模型。

`engine.py` 组织 TokenizerManager、Scheduler、Detokenizer 等 runtime process；`scheduler.py` 的 `run_scheduler_process` 创建 Scheduler 并运行事件循环，见 [`scheduler.py:5876-5958`](../repos/sglang/python/sglang/srt/managers/scheduler.py)。

### 5.2 Scheduler 是运行时批处理状态机

Normal loop 主流程是 ingest requests → 选下一批 → run_batch → process_batch_result，见 [`scheduler.py:1907-1939`](../repos/sglang/python/sglang/srt/managers/scheduler.py)。Overlap loop 把上批结果处理与当前 batch 的 GPU forward 尽量重叠，用队列、stream/event/WAR barrier 维护依赖关系，见 [`scheduler.py:1941-2014`](../repos/sglang/python/sglang/srt/managers/scheduler.py)。

`get_next_batch_to_run` 融合上轮 prefill/extend 与 running decode，处理 chunked request、HiSparse、DP attention 等分支，再决定加新 prefill 还是跑 decode，见 [`scheduler.py:3643-3785`](../repos/sglang/python/sglang/srt/managers/scheduler.py)。因此 continuous batching 的核心不只是「有 batch」：它依赖 running batch 生命周期、chunked prefill 的搬运、KV 状态变化以及输出状态回流。

### 5.3 Prefix cache 是调度决策的一等输入

SGLang 的 `RadixCache` 基于 token key 查最长公共前缀，且 key 可用 `extra_key`/`cache_salt` 做 namespace 隔离，见 [`radix_cache.py:59-82`](../repos/sglang/python/sglang/srt/mem_cache/radix_cache.py)、[`radix_cache.py:355-413`](../repos/sglang/python/sglang/srt/mem_cache/radix_cache.py)。插入、引用保护和淘汰有各自显式操作（`radix_cache.py:415-435, 497-615`）。prefix match 会受到 token ID、page alignment、命名空间和具体 allocator/cache 类型影响，不是对原始文本做任意共享。

新 prefill batch 构造时 `SchedulePolicy.calc_priority` 调整 waiting queue，`PrefillAdder` 将 prefix hit、缺失 token、chunk 大小、KV memory budget 等纳入准入。见 [`scheduler.py:3807-3926`](../repos/sglang/python/sglang/srt/managers/scheduler.py) 与 [`schedule_policy.py:621-740`](../repos/sglang/python/sglang/srt/managers/schedule_policy.py)。策略有 FCFS、cache-aware 等变体；某些大队列情形会退回 FCFS，不能概括成始终以最长缓存前缀优先。

### 5.4 哲学归纳

- 将请求协议化后通过 runtime 消息和 RID 状态关联，让 HTTP、tokenization、调度、模型执行不互相阻塞。
- 把 prompt prefix 命中与新请求准入/优先级、memory budget 直接联系，强调复用已经算过的上下文状态。
- 把 continuous batching、chunked prefill、prefill/decode 交错和 overlap 作为 scheduler 主循环的中心问题。
- 保留多种政策与 runtime 扩展路径（不同 prefix policy、HiCache、disaggregation、LoRA、speculative、multimodal、DP/PP），代价是 scheduler 复杂、配置分支众多。

## 6. 横向比较表

| 维度 | vLLM V1（本 commit） | SGLang SRT（本 commit） |
|---|---|---|
| 对外 server | `launchers` 组 app/engine 生命周期；OpenAI `api_server.py` 是 deprecated shim | FastAPI `http_server.py` 是主要 HTTP 入口，含 native 与多协议 routes |
| 前端抽象 | AsyncLLM / EngineClient，输出按请求 collector 汇聚 | TokenizerManager + RID state + typed IO/socket 回流 |
| 主调度抽象 | Scheduler 每步安排 token progress 与显式 token budget | Scheduler 持 running/waiting batches，衔接 prefill/decode/chunks |
| Prefix reuse | hash 完整 token blocks + BlockPool/refcount；cache 可与 connectors 协作 | Radix longest-prefix cache；命中信息进入 policy/admission/memory budget |
| KV 资源控制 | KVCacheManager façade、cache groups/specs、block allocation；失败交由 Scheduler defer/preempt | allocator/pool + prefix tree + PrefillAdder；缓存命中/预算直接影响 prefill admission |
| 批处理 | token-budgeted unified scheduling；可 batch queue/async schedule | continuous batching 主循环，可 normal/overlap loop，混合 running decode 与新 prefill |
| 内存不足 | Scheduler 在无法 allocate slots 时按 policy preempt running request | scheduler/prefill adder 受内存预算约束，可延迟/分块/优先级处理；具体 retraction 分支依配置 |
| 扩展焦点 | execution、KV spec/connector、Executor/EngineCore、兼容多任务模型的模块边界 | runtime scheduler、prefix cache、HiCache/disaggregation、agentic/多模态/策略集成 |
| 主要复杂性 | 多层进程/抽象和 connector/cache-group 交互；追踪一次请求要跨 EngineClient/Core/Scheduler | 大 Scheduler 及 mixin/component 很多；批状态、cache policy、异步 overlap 相互影响 |

## 7. 设计选择的含义（不是绝对优劣）

### 7.1 何时 vLLM 的组织方式更方便理解/扩展

若问题围绕「engine 和 API 是否同进程、worker 如何部署、模型如何执行、KV cache 类型如何注册、异步 RPC 怎么传输」，vLLM 的分层界面更清晰。新执行后端或多种 cache spec 可沿 Executor/KV interface 边界接入。调度器仍很复杂，但组件间接口较明确。

### 7.2 何时 SGLang 的组织方式更突出

若问题围绕「相同前缀请求怎样复用、cache hit 怎样改变 prefill 成本/优先级、running batch 怎么与新 prompt 交错、如何配置调度策略」，SGLang 的 RadixCache → SchedulePolicy/PrefillAdder → Scheduler 流更直接地揭示设计意图。overlap 模式也将 CPU/GPU 同步策略放在 scheduler loop 中显式呈现。

### 7.3 需要避免的误读

1. 不能说 vLLM 不支持 prefix caching、异步或 chunked prefill；其 V1 scheduler 文档直接列出这些兼容性。
2. 不能说 SGLang prefix cache 只缓存字符串；缓存关联的是 KV 索引/节点、命中受 token key/页对齐/命名空间约束。
3. 不能把 vLLM 的 `api_server.py` 当本 commit 的真正服务入口；该文件是 deprecated compatibility shim。
4. 不要把「SGLang cache-aware」等同于永远 LPM 优先；队列压力、策略、内存预算会改变实际排序。
5. 不应宣称任一框架所有部署都启用 overlap、多进程 EngineCore、P/D disaggregation 或某种 cache backend；这些均受配置影响。
6. Preemption/cancel 不等于 GPU 上当刻立刻中止；in-flight forward、KV transfer、引用保护和结果清理会影响真正资源释放时点。
7. 此文只看 server/engine 主路径，不代表已逐项追踪所有 OpenAI route、错误码、流式协议、调度插件或硬件后端。

## 8. 多种 GPU / 加速器的适配方案

> 这里的“多种 GPU”指 NVIDIA、AMD、Intel 等不同厂商设备以及 TPU/NPU/CPU 等计算平台的后端适配；它不同于一台机器上多卡 tensor parallel，也不意味着各平台具有相同的模型、算子或性能覆盖。以下结论限定在本文开头列出的两个 commit。

### 8.1 共同的分层：平台识别 → 设备能力 → 子系统后端 → 算子实现

两项目都没有试图把所有硬件藏在一个“通用 GPU kernel”后面，而是把硬件差异分层处理：

1. **平台识别**：启动时判定当前运行时/设备平台，实例化 platform object；外部硬件厂商可通过插件扩展。
2. **设备与能力抽象**：统一设备类型、设备属性、内存/同步/分布式操作和能力查询，同时允许按设备代际（例如 CUDA compute capability）进一步分支。
3. **Serving 子系统选择**：针对 attention、graph/capture、KV pool/allocator、quantization、通信等关键子系统选择相应实现。
4. **算子级 dispatch**：同一逻辑运算可以有多个实现；依据平台、设备能力、数据类型、shape、依赖是否安装等条件筛选可用实现，必要时走较通用实现或报不支持。

因此适配成本并不止是能否 `torch.cuda.is_available()`：真正决定“某模型在某张卡上能否运行/跑多快”的，是目标平台的算子、模型路径、KV 格式、collective backend 和构建依赖是否闭环。

### 8.2 vLLM：Platform 插件作为硬件边界，内核按平台/能力组合选择

vLLM 在 `vllm.platforms` 维护平台抽象与运行时发现逻辑。内置检测覆盖 CUDA、ROCm、XPU、CPU、TPU；当前快照的 README 还列出 Intel Gaudi、IBM Spyre、Huawei Ascend、Rebellions、Apple Silicon、MetaX 等 out-of-tree（OOT）硬件插件示例。发现插件时只允许一个 platform plugin 被激活；显式 CPU target 可优先覆盖宿主机上的 accelerator 检测。每个平台类提供设备类型、设备能力、内存/通信、默认 attention/backend 和编译相关策略，平台抽象详见 [vLLM platform interface](../repos/vllm/vllm/platforms/interface.py) 与 [platform discovery](../repos/vllm/vllm/platforms/__init__.py)。

落到关键算子上，vLLM 使用**平台族候选表 + 实现自检/可用性筛选**而非单一跨厂商实现。例如 [linear kernel selector](../repos/vllm/vllm/model_executor/kernels/linear/__init__.py) 按 `PlatformEnum.CUDA/ROCM/XPU/CPU` 为 FP8、INT8、低比特线性层等登记不同候选及优先级：CUDA 可选 CUTLASS、FlashInfer、Marlin 等；ROCm 有 AITER/ROCm 专用实现并在部分算子使用 Triton；XPU 与 CPU 则走各自专用 kernel 或较通用实现。Attention 也有 [backend registry](../repos/vllm/vllm/v1/attention/backends/registry.py)，其中分别登记 FlashAttention、FlashInfer、ROCm AITER、XPU/CPU 等实现；平台 backend 再按照设备代际、dtype、模型特征和配置来判定优先级或可用性。

**设计特点**：vLLM 的主干执行/调度抽象尽量保持一致，把硬件差异收敛在 platform implementation、attention/linear kernel registry、worker/executor 等边界。OOT 插件可减少将厂商专属代码直接并入核心的压力，但平台插件“能被发现”本身不等于所有模型和所有 kernel 都自动可用；部分插件还需要独立编译产物、依赖栈或对特定模型/算子做适配。

### 8.3 SGLang：SRTPlatform 提供运行时工厂，同时以硬件目录和算子分派接入专用实现

SGLang SRT 通过 [平台发现模块](../repos/sglang/python/sglang/srt/platforms/__init__.py) 懒加载一个 `current_platform`。无插件时按环境识别内置 CUDA、ROCm、NPU、XPU 或 CPU；`SGLANG_PLATFORM` 可指定 entry-point 插件，若自动发现到多个 OOT platform 则要求显式选择。插件需实现 [SRTPlatform interface](../repos/sglang/python/sglang/srt/platforms/interface.py)，向 SRT 提供 attention backend、graph runner、MHA/MLA/DSA KV pool、paged allocator、compile backend、量化配置及能力标志等工厂/能力接口；[DeviceMixin](../repos/sglang/python/sglang/srt/platforms/device_mixin.py) 统一设备查询和基础设备操作。

SGLang 同时允许平台差异下沉到 `hardware_backend` 及算子实现目录，而不强制所有厂商都挤进一份通用代码。例如本 checkout 存在 Ascend NPU 专用 attention/backend 文件、XPU 后端、MUSA 后端和 MLX 路径。算子侧，[SGLang kernel organization 说明](../repos/sglang/python/sglang/kernels/README.md) 描述了 `BaseFusedOp` 的两条正交维度：一条是实现来源（Triton、FlashInfer、AITER、Torch-NPU 等），另一条是设备平台路径（CUDA/HIP/NPU/XPU/MUSA/CPU/OOT）；能力声明、平台信息和调用 shape/dtype gate 参与选择，不可用时可退到 native Torch 路径。要注意此处文档描述的是相应 fused-op 路由，不应推导为 SGLang 所有 kernel 都有完整的跨平台 fallback。

SGLang README 当前快照列举 NVIDIA、AMD、Google TPU、Intel GPU/CPU、Apple Silicon、Huawei Ascend、Moore Threads 等硬件；但其中 TPU 有 SGL-JAX/SGL-torchtpu 等独立生态路径，Apple Silicon 包含 Metal/MLX 路径，不能一概理解成与 CUDA SRT 同一套 runner 的等价实现。硬件列表是项目支持面说明，不代表每一型号、模型、量化格式和功能都通过同一测试矩阵。

**设计特点**：SGLang 的 platform object 不只标记 device type，也参与为 SRT subsystem 提供实现工厂；部分平台拥有相对独立的 `hardware_backend/<vendor>` 代码树。与此同时，部分旧调用点仍保留 `is_cuda()`/`is_hip()` 等平台条件分支，平台接口注释也指出有些方法尚未接管所有核心调用。因此它呈现“公共 SRT 接口 + 按平台子系统实现 + 逐步迁移中的历史分支”并存，而非完全插件化的统一层。

### 8.4 简单对照与选型含义

| 维度 | vLLM（该快照） | SGLang（该快照） |
|---|---|---|
| 平台扩展入口 | `vllm.platforms.Platform` + platform plugin entry point | `SRTPlatform` / `DeviceMixin` + `sglang.srt.platforms` entry point |
| SRT/engine 侧接口重点 | backend/设备策略与执行栈集成，执行器抽象保持主干一致 | 为 attention、graph runner、KV pool/allocator、compile 等提供运行时工厂 |
| 算子选择样式 | 按平台枚举的候选列表、支持性判断和优先级；Attention backend registry | fused-op 能力声明与平台方法并用；平台专用目录覆盖特殊子系统 |
| 厂商专用代码的放置 | 内置平台目录、平台专用 kernel，或 OOT 插件 | `hardware_backend/<vendor>`、platform class、算子 backend 或 OOT 插件 |
| 应如何解读“支持” | platform/plugin 被识别只是第一层，继续核实模型与 kernel/backend 支持 | README 硬件列表覆盖面较宽，但各平台可能走不同 runtime / runner，须核实具体部署路径 |

**结论**：二者的共同策略是“统一 serving 上层 + 按平台替换关键计算/运行时部件”，而不是保证一个 GPU kernel 二进制在所有卡上通吃。vLLM 的可见重点是平台抽象与显式 backend/候选选择；SGLang 的可见重点是平台工厂接口与 vendor hardware backend 的组合。实际评估适配时建议按“硬件型号/驱动栈 → PyTorch/runtime → 模型架构 → attention/MoE/quant kernel → KV/graph → TP collective → CI/benchmark”逐项确认；不要只按 README 的平台名或插件检测结果判断生产可用性。

### 8.5 相关源码入口

- vLLM：[平台接口](../repos/vllm/vllm/platforms/interface.py)、[平台发现/插件](../repos/vllm/vllm/platforms/__init__.py)、[CUDA platform](../repos/vllm/vllm/platforms/cuda.py)、[ROCm platform](../repos/vllm/vllm/platforms/rocm.py)、[XPU platform](../repos/vllm/vllm/platforms/xpu.py)、[Attention registry](../repos/vllm/vllm/v1/attention/backends/registry.py)、[linear kernel 候选表](../repos/vllm/vllm/model_executor/kernels/linear/__init__.py)。
- SGLang：[平台发现/插件](../repos/sglang/python/sglang/srt/platforms/__init__.py)、[SRTPlatform 接口](../repos/sglang/python/sglang/srt/platforms/interface.py)、[DeviceMixin](../repos/sglang/python/sglang/srt/platforms/device_mixin.py)、[启动平台 hook](../repos/sglang/python/sglang/srt/arg_groups/platform_hook.py)、[kernel backend 说明](../repos/sglang/python/sglang/kernels/README.md)、[NPU backend 目录](../repos/sglang/python/sglang/srt/hardware_backend/npu/attention)。

## 9. 推荐复盘路线

1. 先从 API 入口跟到请求对象：vLLM `launchers/api_server/entry.py`；SGLang `http_server.py` + `tokenizer_manager.py`。
2. 再看跨进程传输与输出复用：vLLM `core_client.py` / `async_llm.py`；SGLang IO structs、TokenizerManager handle loop 和 detokenizer。
3. 然后手工跟踪一次短 prompt：vLLM `EngineCore.add_request → Scheduler.schedule → allocate_slots → execute_model → update_from_output`；SGLang `handle_generate_request → waiting_queue → PrefillAdder → run_batch → process_batch_result`。
4. 最后追 cache：vLLM `KVCacheManager → BlockPool`；SGLang `match_prefix → PrefillAdder → RadixCache insert/evict/refcount`。
5. 对照 metrics、abort、prefill chunk、OOM/preempt、spec decode 和 P/D disaggregation 作为反例路径。

## 9. 文件链接与可复查入口

### vLLM

- [FastAPI server 兼容入口（deprecated shim）](../repos/vllm/vllm/entrypoints/openai/api_server.py)
- [API server app/engine 构造](../repos/vllm/vllm/entrypoints/launchers/api_server/entry.py)
- [AsyncLLM](../repos/vllm/vllm/v1/engine/async_llm.py)
- [EngineCore](../repos/vllm/vllm/v1/engine/core.py)
- [EngineCoreClient](../repos/vllm/vllm/v1/engine/core_client.py)
- [V1 Scheduler](../repos/vllm/vllm/v1/core/sched/scheduler.py)
- [KVCacheManager](../repos/vllm/vllm/v1/core/kv_cache_manager.py)
- [BlockPool](../repos/vllm/vllm/v1/core/block_pool.py)

### SGLang

- [HTTP server](../repos/sglang/python/sglang/srt/entrypoints/http_server.py)
- [Engine process orchestration](../repos/sglang/python/sglang/srt/entrypoints/engine.py)
- [TokenizerManager](../repos/sglang/python/sglang/srt/managers/tokenizer_manager.py)
- [Scheduler](../repos/sglang/python/sglang/srt/managers/scheduler.py)
- [Schedule policy / PrefillAdder](../repos/sglang/python/sglang/srt/managers/schedule_policy.py)
- [RadixCache](../repos/sglang/python/sglang/srt/mem_cache/radix_cache.py)
- [TP worker](../repos/sglang/python/sglang/srt/managers/tp_worker.py)

## 10. 本次工作状态

- 已确认两个仓库 HEAD 未变，且 working tree clean。
- vLLM `git remote origin`：`https://github.com/vllm-project/vllm.git`。
- SGLang `git remote origin`：`https://github.com/sgl-project/sglang.git`。
- 已将 server/engine/scheduler/cache 相关 sparse-checkout 路径补齐。
- 已阅读主要入口、AsyncLLM/EngineCore、Scheduler/KV manager、SGLang HTTP/TokenizerManager/Scheduler/SchedulePolicy/RadixCache 的关键代码。
- 未安装依赖、未启动服务、未运行 GPU workload 或 benchmark；这是静态源码分析。
