# DS 学习手册

以 **V4 Flash** 建立一条完整计算主线，再比较 **V4 Pro / V4.1 Flash**。新版将共同机制合并、统一张量符号，并把动画放进对应章节。

## 从这里开始

- **[在线阅读正文](ds-handbook.md)**：按顺序编排的 11 章 Markdown。
- **[下载交互手册](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/ds-handbook.html)**：单文件 HTML，下载后双击打开；8 组交互计算可离线播放，不含视频。
- [个人工作笔记](notes/)：与模型手册分开保存。

## 新版框架

| 章节 | 要解决的问题 |
|---|---|
| 01 阅读地图与张量语言 | token 轴、特征轴和四种权重分别是什么？ |
| 02 整体模型与 mHC | 一个完整 Block 怎样计算和回写残差？ |
| 03 Attention | SWA、CSA、HCA 分别读取哪些 KV？ |
| 04 Compressor | gate 怎样逐维压缩，短期状态保存多久？ |
| 05 Indexer | 点积、头权重、因果 mask 和 top-k 怎样衔接？ |
| 06 MoE | 查表路由、选择偏置、混合系数和专家怎样计算？ |
| 07 MegaMoE | 路由张量怎样派发、执行，再按 token 合并？ |
| 08 量化与缓存 | 逻辑 shape、数值精度与物理字节如何分别计算？ |
| 09 V4.1 增量 | 跨层共享、CED、Engram、DSpark、视觉改变了什么？ |
| 10 生成、服务与训练 | 模型前向怎样放进完整运行流程？ |
| 11 源码与维护 | 到哪里验证结论，以后怎样补充而不重复？ |

版本：2026-10-14。每章包含适用范围、输入输出、计算步骤、示例与固定版本源码链接。动画为缩小后的教学数据，不代表 checkpoint 实际数值或 GPU 性能。

## 获取与更新

```bash
git clone https://github.com/yifeihu0926/llm-learn.git
cd llm-learn
```

已经克隆过：

```bash
git pull --ff-only
```

<details>
<summary>历史资料与旧版入口（备查）</summary>

旧版按不同时间和模型分别生成，存在重复和部分失效的本机链接。日常学习以新版手册为入口，旧版用于追踪历史分析，不继续平行维护多套正文。

- [旧综合版](deepseek-v4-framework.html)
- [旧 Flash](deepseek-v4-flash.html) · [旧 Pro](deepseek-v4-pro.html) · [旧 V4.1](deepseek-v41-flash.html)
- [旧量化专题](deepseek-v4-quantization.html) · [旧 MegaMoE 动画](mega-moe-full.html)
- [第三方组件许可](THIRD-PARTY-NOTICES.md)（旧版字体与公式组件；新版正文与交互不依赖这些组件）

</details>
