# CUDA 卷积融合算子：Conv + Bias + Residual + Activation

资料核查与整理日期：2026-10-15。目标平台：NVIDIA GPU / CUDA。

[下载完整交互阅读版](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/conv-fusion.html) · [下载独立动画](https://github.com/yifeihu0926/llm-learn/raw/refs/heads/main/conv-fusion-animation.html) · [返回资料主页](README.md)

HTML 下载后用浏览器打开；完整阅读版将正文、公式和动画内嵌在同一个文件中，不需要安装依赖或启动服务。GitHub 文件预览页只展示源码，不会执行 HTML。官方参考链接需要联网。

## 1. 算子目标与语义契约

推荐首先固定这个运算：

$$
Y=\operatorname{Act}\left(\alpha\operatorname{Conv}(X,W)+B+\beta Z\right)
$$

B 是按输出通道广播的 bias，Z 是 residual。执行顺序是 Conv → Bias → Residual → Activation。Act(Conv+Bias)+Residual 是另一种算子。即使只交换两次加法，在有限精度下也可能因舍入次序不同而产生差异。

| 项目 | 应写入接口的约定 |
|---|---|
| 卷积 | Conv2D；默认采用 cross-correlation，不翻转权重 |
| 逻辑维度 | X: [N,C,H,W]；W: [K,C/G,R,S]；Y: [N,K,P,Q] |
| 参数 | stride、前后 padding、dilation、groups G；C 和 K 均可被 G 整除 |
| Bias | [K]，沿 batch 和输出空间广播；是否可省略 |
| Residual | 首先明确同形状输入；广播能力作为单独支持项 |
| 缩放 | alpha、beta 的类型与默认值；本动画均为 1 |
| 激活 | Identity/ReLU/SiLU/GELU 等；GELU 需指定近似版本 |
| 精度 | 输入、权重、bias、residual、累加、epilogue、输出分别定义 |
| 内存 | strides、对齐、alias；是否允许 output 覆盖 residual |
| 场景 | 推理或训练；是否要求确定性；workspace 上限 |

输出尺寸：

$$
P=\left\lfloor\frac{H+p_{\mathrm{top}}+p_{\mathrm{bottom}}-d_h(R-1)-1}{s_h}\right\rfloor+1
$$

Q 同理。这里 s_h 是 stride，d_h 是 dilation。不要把 kernel 的空间尺寸 R/S 与 CUDA kernel 程序混淆。

## 2. 数学定义：从一个输出元素开始

以下写成物理 NHWC / KRSC 索引。令 C_g=C/G，K_g=K/G，输出通道 k 对应组 g=floor(k/K_g)。定义：

$$
h=p s_h-p_{\mathrm{top}}+r d_h,\qquad w=q s_w-p_{\mathrm{left}}+s d_w
$$

$$
V_{n,p,q,k}=\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}\sum_{c=0}^{C_g-1}X_{n,h,w,gC_g+c}\,W_{k,r,s,c}
$$

X 的空间坐标越界时贡献为零。然后：

$$
U_{n,p,q,k}=(\alpha V_{n,p,q,k}+b_k)+\beta Z_{n,p,q,k},\qquad Y_{n,p,q,k}=\operatorname{Act}(U_{n,p,q,k})
$$

本动画使用 ReLU，ReLU(u)=max(0,u)。一般算子可扩展 SiLU(u)=u·sigmoid(u)，以及指定版本的 GELU。

### 动画的固定输入与可核对结果

动画范围：N=1，H=W=4，C=K=2，R=S=2，stride=1，padding=0，dilation=1，groups=1，alpha=beta=1。输入是 NHWC，权重 KRSC，输出 NPQK；这里的布局不改变框架逻辑维度的命名。

输入和权重为小整数：

```text
X[0,h,w,c] = 8*h + 2*w + c
j = (r*S+s)*C+c
W[k=0, j] = [1,-1,1,-1,1,-1,1,-1]
W[k=1, j] = [1, 0,0, 0,0, 0,0, 1]
bias = [3,-16]
Z[m] = [m-2, -2*m]
m = 3*p+q
```

输出尺寸是 1×3×3×2，每个输出元素归约 8 项，共 18 个输出、144 次乘加。最终输出：

```text
p=0: [0,0]   [0,0]   [0,0]
p=1: [0,5]   [1,7]   [2,9]
p=2: [3,15]  [4,17]  [5,19]
```

例如 m=4 对应 (p,q)=(1,1)：

```text
输入滑窗逻辑行 A[4] = [10,11,12,13,18,19,20,21]
前四项后的 Acc      = [-2,10]
全部八项后的 Acc    = [-4,31]
加 bias            = [-1,15]
加 Z[4]=[2,-8]     = [1,7]
ReLU               = [1,7]
```

<!-- CONV_ANIMATION -->

## 3. Implicit GEMM：把滑窗映射成 tile 计算

对每个 group 单独考虑：

$$
M_{\mathrm{gemm}}=NPQ,\quad N_{\mathrm{gemm}}=K/G,\quad K_{\mathrm{gemm}}=(C/G)RS
$$

为避免符号冲突，本动画把归约编号写成 j，输出通道仍写作 k。groups=1 时采用 c 最快的编号：

$$
m=(nP+p)Q+q,\qquad j=(rS+s)C+c
$$

$$
A_{m,j}=X_{n,h,w,c},\quad B_{j,k}=W_{k,r,s,c},\quad \mathrm{Acc}_{m,k}=\sum_j A_{m,j}B_{j,k}
$$

本例为 (9×8)·(8×2)→(9×2)。A 是滑窗的逻辑矩阵；B 是 KRSC 权重的逻辑矩阵视图，图上的相邻列不表示真实权重存储中 k 连续。

Implicit GEMM 在访问 tile 时生成地址和边界 predicate，并不先在显存中创建完整 im2col 矩阵。本动画的 j 编号是教学选择；实际实现可采用其他归约遍历顺序，只要 A 与 B 的置换一致。实现背景见 [CUTLASS Implicit GEMM Convolution](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html)。

## 4. CUDA kernel 的计算定义

三个概念需要分开：卷积核 W 是权重；CUDA kernel 是 GPU 执行的程序；MMA 是实现矩阵乘加的一类硬件指令。一个 CTA 是一个线程块，通常由多个线程协作计算输出 tile。

动画使用 TILE_M=4、TILE_N=2、TILE_K=4，只是缩小的逻辑 tile。寄存器面板代表整个 CTA 的逻辑累加器集合，没有指定真实 warp、lane、MMA 尺寸或寄存器分布。

```cpp
// 说明依赖关系的伪代码，不可直接编译
m0 = blockIdx.x * TILE_M;
k0 = blockIdx.y * TILE_N;
Acc = zeros(TILE_M, TILE_N);

for (j0 = 0; j0 < C * R * S; j0 += TILE_K) {
    // 从卷积坐标生成 A，不物化完整 im2col
    As = load_input_tile_with_predicates(X, m0, j0);
    Bs = load_weight_tile_with_predicates(W, j0, k0);
    wait_until_tiles_ready();
    Acc += tile_matrix_product(As, Bs);
    wait_until_tiles_consumed_before_reuse();
}

// 完整归约后执行一次 epilogue
T = (alpha * Acc + broadcast(bias)) + beta * load(residual);
store_valid_outputs(Y, cast_output(activation(T)));
```

本例有 3 个 M tile：m=0…3、4…7、8…11。最后一个仅 m=8 有效；m=9/10/11 的输入、residual 和输出访问都需要屏蔽。通道尾部与 K 尾部在通用实现中也需要正确处理。

两个 K stage 是同一个 CTA 依次累加 j=0…3、4…7，**不是 split-K**。真实优化可使用多级流水线、异步拷贝或架构专用机制，但必须满足加载就绪与缓冲区复用的依赖。同步方式、buffer 生命周期和线程职责随实现选择而定。

## 5. 为什么把 Bias / Residual / Activation 放进 epilogue

理想路径是在卷积累加结果仍可被 kernel 使用时，加载 bias 和 residual，完成激活和输出转换，再写回最终 Y。residual 仍需要读取；收益来自省去中间结果的反复落地及部分 launch 开销。

令输出元素数 E=NKPQ，每元素占 s 字节。忽略 bias、缓存、workspace，且假设中间类型相同，可推导输出链路逻辑流量：

| 路径 | 逻辑流量 |
|---|---:|
| Conv / Bias / Residual / Activation 四段分离 | 8Es |
| 独立 Conv + 一个融合后处理 kernel | 4Es |
| 卷积与全部 epilogue 融合 | 2Es |

这不是 DRAM 实测，也不代表 4 倍加速。融合可能增加寄存器使用、降低占用率，或者限制卷积算法选择。必须保留“最快独立卷积 + 融合后处理”对照。

若采用 split-K，非线性激活必须放在完整归约之后；bias/residual 各加一次。ReLU(-3+4)=1，而 ReLU(-3)+ReLU(4)=4，二者不等价。

## 6. cuDNN 与 CUTLASS 的落地路线

### cuDNN Graph API 基线

旧接口 cudnnConvolutionBiasActivationForward 的公式直接包含 residual：y=act(alpha1·conv+alpha2·z+bias)，但在 cuDNN 9.0 已标记 deprecated。新实现优先使用 Graph API。[旧接口定义](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.0/api/cudnn-cnn-library.html)

Graph 的节点顺序为 Conv → Add(Bias) → Add(Residual) → Activation，按以下流程构建：

```text
validate → build_operation_graph → create_execution_plans
→ check_support → build_plans → 调优 → execute
```

将中间结果设为 virtual，检查目标硬件、dtype、layout、groups、激活的支持范围。一个 graph 可能由一小组 kernel 执行；单次 graph 调用不证明已经单 kernel 融合，需要 profiler 证据。[cuDNN Graph](https://docs.nvidia.com/cudnn/latest/developer/graph-api/)、[Frontend 工作流程](https://docs.nvidia.com/cudnn/latest/developer/overview/)

构建和调优放在初始化或缓存未命中路径；完整操作属性、shape、strides、dtype、数学模式与目标环境用于分派和缓存。运行时复用 plan/workspace。[集成性能指南](https://docs.nvidia.com/cudnn/latest/utilities/framework_integration_performance/)

官方 [fprop.cpp](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/convolution/fprop.cpp) 的 CBR Graph NCHW 包含 Conv → Residual Add → Bias Add → ReLU。功能覆盖相同，但加法顺序不同；对严格中间舍入契约应调整并验证。

### CUTLASS 自定义实现

复用卷积主体，定制带广播和额外输入的 epilogue。关键入口：

- [per-channel bias 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/17_fprop_per_channel_bias/fprop_per_channel_bias.cu)
- [DefaultConv2dFpropWithBroadcast](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h)
- [LinearCombinationResidualBlock](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/thread/linear_combination_residual_block.h)

ResidualBlock 的单 residual 路径结构为 UnaryOp(BinaryOp(ActivationOp(alpha·Acc+Bias),beta·Residual))。对于本目标，应把 ActivationOp 设为 Identity、BinaryOp 设为 Plus、UnaryOp 设为目标激活。仍需核对具体 kernel 接口、类型和 split-K 支持，不能只替换 functor 就假定全部组合可用。

按目标架构选择实现：除旧 Ampere 路线，还应检查 Hopper 的 3.x convolution，以及 [Blackwell convolution 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv)。API 和架构能力以选定 release/tag 的源码为准。[CUTLASS Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)

## 7. 布局、访存与形状分派

优先评估 X=NHWC、W=KRSC、Z/Y=NPQK，连同全模型布局转换成本一起测量。框架的逻辑 NCHW shape 可以采用 channels-last 物理存储；不能只看 shape 字符串判断布局。

Bias 按 k 广播；residual 按输出坐标合并读取。accumulator fragment 与 residual fragment 的坐标必须对应。向量化要同时满足指针、stride、元素对齐和尾部安全。若允许 Y 与 Z 共用内存，应单独验证 alias 和读写时序。

| 形状类别 | 优先研究 |
|---|---|
| 1×1 / stride1 / pad0 | GEMM 式实现与 epilogue |
| 稠密 3×3 | implicit GEMM 与 cuDNN 候选算法 |
| 大 kernel | 不同算法及 workspace 成本 |
| Group conv | 每组规模与尾块利用率 |
| Depthwise | 专用 direct kernel |
| 小 batch / 小输出 | launch 开销和并行度 |
| 大归约 / 少输出 tile | 评估 split-K 及最终归约 |

NVIDIA 的性能资料介绍了 implicit GEMM、布局和 tile 利用率；其中具体性能数字对应文档中的设备和软件环境，不应直接迁移到新 GPU。[Convolution Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)

## 8. 精度、量化和训练

### 浮点精度

FP16/BF16 输入可优先评估 FP32 累加和 FP32 epilogue，最后转回输出类型。与分离算子相比，融合可能减少中间舍入；数学等价不保证逐位一致。固定 TF32、归约、激活近似和中间 cast 策略，再比较误差。[PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

建立小 shape 的 FP64 数学 reference，以及固定框架语义的 dtype reference。报告 max absolute error、相对误差与整体误差统计，零附近不能只看相对误差。

### INT8 / FP8

以对称 INT8 为例，A_k=sum(q_x·q_w,k)，输入 scale 为 s_x，权重每通道 scale 为 s_w,k，residual scale 为 s_z：

$$
U_k=\alpha s_x s_{w,k}A_k+b_k+\beta s_z q_{z,k},\qquad q_y=\operatorname{Quantize}(\operatorname{Act}(U_k),s_y)
$$

不同 scale 的 accumulator 和 residual 不能直接做整数加法。还需定义 bias 表示、zero-point 修正、舍入、饱和和 Q/DQ 边界；FP8 需要明确格式及缩放规则。[TensorRT Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)

### 训练反向

令 U=alpha·Conv+B+beta·Z，D_U=D_Y⊙Act'(U)：

$$
D_Z=\beta D_U,\quad D_B[k]=\sum_{n,p,q}D_U[n,p,q,k]
$$

$$
D_X=\alpha\operatorname{ConvBackwardData}(D_U,W),\qquad D_W=\alpha\operatorname{ConvBackwardFilter}(X,D_U)
$$

ReLU 需要明确零点导数，并保存 mask 或等价信息；SiLU/GELU 通常保存激活输入或重计算。前向节省的显存应与反向辅助信息成本一起评估。前向融合不表示整个反向可以合成一个 kernel。

## 9. 接口与框架集成

```cpp
Status fused_conv_bias_residual_act(
    TensorView x, TensorView weight,
    OptionalTensorView bias, OptionalTensorView residual,
    TensorView output, ConvParams conv,
    ActivationParams activation, Scalar alpha, Scalar beta,
    PrecisionPolicy precision,
    void* workspace, size_t workspace_bytes, cudaStream_t stream);
```

定义 shape/stride/dtype/groups/device/alias 校验，workspace 查询、分派、缓存、精度与确定性策略，提供正确回退路径。使用调用方 stream，避免每次调用重新分配和调优。

PyTorch 自定义算子需要 schema、mutation/alias、FakeTensor，训练还需 autograd。opcheck 验证注册契约；前向数值和梯度需分别验证。[Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)

## 10. 验证与性能测量

正确性覆盖 1×1/3×3/大 kernel，stride/dilation/非对称 padding，groups/depthwise，奇数高宽，非对齐通道，bias/residual 可选，激活阈值附近、相消、大幅值，以及承诺支持的 dtype/stride/alias/stream。若支持 split-K，应测试每种归约路径；训练需覆盖全部梯度。NaN/Inf、无效 shape 和零元素输入要有明确契约。

真实 CUDA 实现使用 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) 检查越界、未初始化读取、竞争和同步问题。

性能至少比较：框架 eager、高性能独立 Conv+融合后处理、cuDNN Graph、自定义 CUTLASS。若部署使用 TensorRT，也比较实际引擎；支持局部融合不保证完整四段必然融合。[TensorRT Fusion Catalog](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/performance/fusion-catalog.html)

构建/JIT/调优与稳定执行分开；预热后在相同 stream 上用 CUDA events 计时，多轮重复。记录端到端延迟、布局转换、kernel 数量、workspace 和缓存条件；公平比较普通 launch 与 CUDA Graph replay。用 Nsight Systems 查看执行链，Nsight Compute 查看 Tensor Core、访存、寄存器、occupancy 和 roofline。不要把 profiler 的宿主计时直接当成正常运行延迟。[Nsight Compute Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)

## 11. 推荐实施顺序与证据边界

先固定完整支持契约，再建立 cuDNN 四功能基线；自定义路径优先从稠密 Conv2D、FP16/BF16、FP32 累加、channels-last、同形 residual 开始，随后按真实 workload 扩展 groups、depthwise、量化和训练。参数调优需提供 GPU 型号、主要 shape、激活和训练/推理场景。

本资料交付的是设计调研、公式和计算动画，尚未实现或执行生产 CUDA kernel，也没有 GPU 性能实测。动画使用 JavaScript Number 与小整数，不模拟 FP16/BF16 舍入；逻辑 tile 不是硬件线程布局。数学与全部输出可独立复算，不能用浏览器动画检查代替 GPU 数值或性能验证。

公开链接中的 latest/main 会变化；实现时应另行锁定 cuDNN/CUTLASS/CUDA 版本和源码提交。HTML 公式由已有 MathJax 3.2.1 工具预先生成 SVG，运行时无需联网；许可见仓库 [第三方组件说明](THIRD-PARTY-NOTICES.md) 和 [MathJax 许可](MathJax-LICENSE.txt)。
