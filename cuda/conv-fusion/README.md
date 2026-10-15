# 卷积融合算子

[返回 CUDA 目录](../README.md) · [返回资料主页](../../README.md)

- **[在线阅读调研与数学定义](conv-fusion.md)**：算子契约、implicit GEMM、CUDA tile、融合 epilogue、精度与验证。
- **[下载完整交互阅读版](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/cuda/conv-fusion/conv-fusion.html)**：正文、公式和三种动画内嵌在单文件中，下载后用浏览器打开。
- [独立计算动画](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/cuda/conv-fusion/conv-fusion-animation.html)：滑窗乘加、矩阵映射、CUDA tile 与 epilogue。

整理日期：2026-10-15。面向 NVIDIA GPU / CUDA；支持动画播放、暂停、单步和进度拖动。教学模型使用小整数，尚无生产 CUDA kernel 或 GPU 性能实测。
