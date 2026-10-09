# DeepSeek V4 / V4.1 模型学习文档

按“完整架构 → 特殊 Block → 具体算子”展开，覆盖本地参考前向的投影、归一化、mHC、Attention、Indexer/selector、MoE、量化、缓存、通信、辅助预测与输出控制。每张机制图配有公式、张量形状与解析。

## 文档入口

| 文档 | 图示与交互 | 下载离线 HTML |
|---|---|---|
| V4 Flash | 9 张机制图；SWA/CSA/HCA 三组动画；完整 MTP | [deepseek-v4-flash.html](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/deepseek-v4-flash.html) |
| V4 Pro | 9 张机制图；SWA/CSA/HCA 三组动画；本模型配置与 MTP | [deepseek-v4-pro.html](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/deepseek-v4-pro.html) |
| V4.1 Flash | 14 张机制图；40 层共享状态交互；Engram、视觉塔与 DSpark | [deepseek-v41-flash.html](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/deepseek-v41-flash.html) |
| 三模型综合对照 | 全局架构、机制比较、预算、推理框架与训练 | [deepseek-v4-framework.html](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/deepseek-v4-framework.html) |

下载后用浏览器打开 HTML。图形、公式、字体和动画均已内嵌，无需安装依赖或启动服务；阅读正文不需要联网，源码与资料链接需要联网。综合版中可以跳转到三份独立版，将四个 HTML 下载到同一目录即可。

## 资料范围

文档版本：2026-10-10。推理文件链接固定到与本地内容逐字节一致的官方提交；vLLM 固定为 `c91dccc08044b1269f351ceec13070fe8f075e10`，SGLang 固定为 `bb9a820f09f97480bfc6d07564fb2b691b8857a3`，Transformers 使用 5.18.0。

参数量按原始构造器 shape 统计，缓存按明确的 BF16 数组或生产打包格式计算。训练设置采用官方报告披露。CPU 边界与公式检查不代表完整权重质量、GPU kernel 数值或吞吐测量。

字体与公式组件许可见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
