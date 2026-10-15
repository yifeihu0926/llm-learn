# 学习资料库

按主题归档的研究资料、技术手册、计算动画与个人笔记。

## 从这里开始

| 主题 | 入口 | 内容 |
|---|---|---|
| DeepSeek | [主题目录](deepseek/README.md) · [当前手册正文](deepseek/ds-handbook.md) | V4 / V4.1、计算主线、源码参考与历史资料 |
| CUDA 算子 | [卷积融合专题](cuda/conv-fusion/README.md) | Conv + Bias + Residual + Activation，数学定义与 kernel 动画 |
| 个人笔记 | [工作笔记](notes/) | 独立保存的日常记录 |
| 第三方许可 | [组件说明](licenses/THIRD-PARTY-NOTICES.md) | 字体和公式组件许可 |

## 目录结构

```text
deepseek/
  ds-handbook.md / .html    当前学习主线
  reference/                MegaMoE 源码、vLLM 算子参考
  assets/                   当前手册配图
  archive/                  历史模型文档、分析、动画与配图
cuda/
  conv-fusion/              卷积融合调研与计算动画
notes/                      个人工作笔记
licenses/                   第三方组件说明与许可
```

## 离线阅读与动画

- [下载 DeepSeek 交互手册](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/deepseek/ds-handbook.html)
- [下载卷积融合完整交互版](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/cuda/conv-fusion/conv-fusion.html)

下载 HTML 后用浏览器打开；GitHub 的 HTML 文件预览只展示源码。单文件阅读版的正文和动画可离线运行，外部参考链接需要联网；旧版配图分离的材料建议下载整个仓库。

## 获取与更新

```bash
git clone https://github.com/yifeihu0926/llm-learn.git
cd llm-learn
```

已有本地副本使用 `git pull --ff-only`。归档日期：2026-10-15；原材料全部保留，旧路径的历史版本可通过 Git 提交记录查看。以后新增材料放入相应主题目录，并更新该目录的 README。
