# TX2 Deploy Guide

在 NVIDIA Jetson TX2 上从零编译并运行一个 **C++ + TensorRT** 推理工程的完整示例。

仓库里既有说明文档，也有可以直接跑的源码——照着做，10 分钟能跑通第一个推理。

## 里面有什么

| 目录 | 内容 |
|---|---|
| `TX2从零部署.md` | 完整流程文档：环境检查 → 模型准备 → 编译 → 建引擎 → 推理 → 排错 |
| `cpp/` | TensorRT 推理的 C++ 最小工程（约 400 行），含 CMake 配置 |
| `python/` | 模型导出与自检脚本 |

C++ 示例做的是 **ResNet18 图像分类**：读图 → 预处理 → TensorRT 推理 → 打印 top-5。代码结构与文本检测、缺陷检测、特征提取各类模型通用，换成自己的 ONNX 只需改预处理和后处理。

## 环境

- 设备：Jetson TX2（或其它 Jetson，除依赖路径外代码通用）
- 系统：JetPack 4.6.x / L4T R32.x
- 组件：CUDA 10.2、TensorRT 8.2、OpenCV 4.x、g++ 7.5、CMake 3.10+

更高版本的 JetPack 也能用，只是文中标注的 API 差异要反过来看（见文档「报错对照」）。

## 快速验证

### 1. PC 上导出模型

```bash
cd python
pip install torch torchvision onnx onnxruntime numpy
python export_onnx.py --out resnet18.onnx
python check_onnx.py --model resnet18.onnx
```

`check_onnx.py` 会检查算子兼容性（比如 TRT 8.2 不支持动态 K 的 TopK），并打印 ORT 的 top-5，留作后续对比。

### 2. 传文件到板子

```bash
scp -r cpp python tx2@ubuntu.local:/home/tx2/trt-demo/
scp resnet18.onnx tx2@ubuntu.local:/home/tx2/trt-demo/
```

### 3. 板子上建引擎

```bash
cd ~/trt-demo
/usr/src/tensorrt/bin/trtexec \
  --onnx=resnet18.onnx --fp16 --workspace=512 \
  --saveEngine=resnet18.engine
```

### 4. 板子上编译并推理

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

./trt_demo --engine ~/trt-demo/resnet18.engine --image test.jpg
```

期望输出：

```
[TRT] engine=/home/tx2/trt-demo/resnet18.engine
[TRT] input  'input' dims=[1,3,224,224]
[TRT] output 'output' dims=[1,1000]
[warmup] 210.3 ms
[perf] 稳态 88.5 ms/张 (11.3 fps)

Top-5:
  class 281   prob 0.9312
  class 285   prob 0.0413
  ...
```

`class` 是 ImageNet 的类别编号，281 对应猫。换上自己的图、对比 ORT 的 top-5，就完成了精度校验。

## 换成自己的模型

三步：

1. **导出**：把 `python/export_onnx.py` 里的 `resnet18` 换成你的模型，固定输入尺寸
2. **预处理**：改 `cpp/src/main.cpp` 里的 `preprocess()`——尺寸、通道顺序、归一化参数
3. **后处理**：改 `main()` 里对 `output` 的解读（分类是 softmax，检测要解析框，分割要 reshape 成掩膜）

TensorRT 的部分（`trt_engine.cpp`）不用动——引擎加载、binding 绑定、显存管理、异步执行都是通用的。

**多输入/多输出模型**只需要改 `trt_engine.cpp` 里 `load()` 的 binding 扫描逻辑，`infer()` 的 bindings 数组已按 `getNbBindings()` 动态分配，不会越界。

## 常见问题

**编译报 `NvInfer.h: No such file or directory`**

头文件在 `/usr/include/aarch64-linux-gnu/`，不在 `/usr/include` 根下。CMake 里的 `TRT_INC_HINTS` 已包含该路径，如果还找不到，手动指定：

```bash
cmake .. -DNVINFER_INCLUDE_DIR=/usr/include/aarch64-linux-gnu
```

**`fatal error: filesystem: 没有那个文件或目录`**

gcc 7.5 没有 C++17 的 `<filesystem>`。本仓库代码未使用该头文件；若你的工程需要，改用 `<experimental/filesystem>` 并链接 `stdc++fs`。

**建引擎卡住不动**

先用 `watch -n 5 'ps -o pid,etime,time,%cpu,stat -C trtexec'` 看 CPU 时间涨不涨。不涨就是死锁，把 `--workspace` 从 1024 降到 512 重试。

更多问题见 `TX2从零部署.md` 的「五个必看的坑」与「报错对照」两节。

## 说明

示例模型用 torchvision 的 ResNet18，权重首次运行自动下载。仓库不含任何预训练模型或引擎文件——引擎必须在目标设备上构建，跨设备拷贝会因 TensorRT 版本或 GPU 架构不同而失效。

---

> 本文档与示例代码基于实际部署过程整理，环境版本、模型结构、命令参数因项目而异，**具体操作请结合自己的实际情况调整，仅供参考**。
