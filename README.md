# TX2 Deploy Guide

在 NVIDIA Jetson TX2 上从零编译并运行 **C++ + TensorRT** 推理工程的完整示例。

仓库里既有说明文档，也有能直接跑的源码：**编译一次，之后换任何 ONNX 模型只需要换 engine 文件和归一化参数**，不用改代码。

## 里面有什么

| 目录 | 内容 |
|---|---|
| `TX2从零部署.md` | 完整流程文档：环境检查 → 模型准备 → 编译 → 建引擎 → 推理 → 排错 |
| `实验报告.md` | 性能实验记录：怎么测延迟、怎么看瓶颈、踩过哪些坑 |
| `cpp/` | 通用 TensorRT 推理工具（约 500 行）+ CMake 配置 + 建引擎脚本 |
| `python/` | 模型导出与自检脚本 |

`cpp/` 下的 `trt_infer` 是一个**不绑定具体模型**的命令行工具：

- 输入尺寸、通道数从 engine 里自动读，不写死 224
- 归一化参数（mean/std）可从命令行覆盖
- 分类模型的输出自动打印 top-5；其它形状打印统计量和前几个值
- 支持单图、整个目录批量

## 环境与依赖

**板子上（跑推理）** —— 这些 JetPack 镜像里都有，一般不用装：

- 设备：Jetson TX2（其它 Jetson 除依赖路径外代码通用）
- 系统：JetPack 4.6.x / L4T R32.x
- 组件：CUDA 10.2、TensorRT 8.2、OpenCV 4.x、g++ 7.5、CMake 3.10+

自检一条命令：

```bash
head -1 /etc/nv_tegra_release && nvcc --version | grep release && \
ls /usr/include/aarch64-linux-gnu/NvInfer.h && pkg-config --modversion opencv4
```

**PC 上（导出模型）** —— Python 依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

**C++ 侧（编译推理程序）** —— 不需要包管理器，依赖由 `cpp/CMakeLists.txt` 自动查找：

| 依赖 | 用途 | 头文件 / 库 | JetPack 里叫什么 |
|---|---|---|---|
| TensorRT | 加载 engine、执行推理 | `NvInfer.h` / `libnvinfer.so` | `tensorrt` |
| CUDA Runtime | 显存分配、异步执行、计时 | `cuda_runtime.h` / `libcudart.so` | `cuda` |
| OpenCV | 读图、resize、色彩转换 | `opencv2/opencv.hpp` / `libopencv_*.so` | `opencv` |

这三个在 JetPack 里都是**预装**的，重新刷机后仍然存在，正常不用装。万一被删了：

```bash
sudo apt-get install -y libopencv-dev
# TensorRT / CUDA 一般随 JetPack 提供，不建议用 apt 单独装，版本容易错
```

编译时 CMake 会打印找到的路径，出现下面四行就说明依赖齐了：

```
-- Found OpenCV: /usr (found version "4.1.1")
-- NvInfer.h: /usr/include/aarch64-linux-gnu
-- nvinfer: /usr/lib/aarch64-linux-gnu/libnvinfer.so
-- cudart : /usr/local/cuda-10.2/lib64/libcudart.so
```

如果 `NvInfer.h` 那行报 `FATAL_ERROR`，手动指定：

```bash
cmake .. -DNVINFER_INCLUDE_DIR=/usr/include/aarch64-linux-gnu
```

`nvonnxparser` 只在「用 C++ 直接从 ONNX 建引擎」时才需要；本仓库统一用 `trtexec` 建引擎，所以代码里不依赖它。

更高版本的 JetPack 也能用，只是文中标注的 API 差异要反过来看（见文档「报错对照」）。

## 三步跑起来

### 1. 板子上编译

```bash
scp -r cpp tx2@ubuntu.local:/home/tx2/trt-demo/

ssh tx2@ubuntu.local
cd ~/trt-demo/cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

编译产物是 `build/trt_infer`，这一步不依赖任何模型文件。

### 2. 准备一个 engine

有现成 ONNX 的，直接建引擎：

```bash
cd ~/trt-demo/cpp
./scripts/build_engine.sh ~/models/your_model.onnx fp16 512
```

没有 ONNX 的，先用示例模型跑通流程（在 PC 上导出，再传到板子）：

```bash
# PC 上
cd python
pip install -r requirements.txt
python export_onnx.py --out resnet18.onnx
python check_onnx.py --model resnet18.onnx

scp resnet18.onnx tx2@ubuntu.local:/home/tx2/models/

# 板子上
cd ~/trt-demo/cpp
./scripts/build_engine.sh ~/models/resnet18.onnx fp16 512
```

### 3. 推理

```bash
./build/trt_infer --engine ~/models/resnet18.fp16.engine --image cat.jpg
```

期望输出：

```
引擎已加载: 输入 3x224x224

[1/1] cat.jpg
  耗时: 预热 210.3 ms | 稳态 88.5 ms (11.3 fps)
  输出形状: [1, 1000]  共 1000 个值
  Top-5:
    class 281    prob 0.9312
    class 285    prob 0.0413
```

`class 281` 对应 ImageNet 的猫。批量跑一个目录：

```bash
./build/trt_infer --engine ~/models/resnet18.fp16.engine --dir ./images --repeat 10
```

## 换成自己的模型

`trt_engine.cpp` 不用动（引擎加载、binding、显存管理都是通用的），`main.cpp` 的预处理也通用。要改的只有两处：

**1. 输入尺寸** —— 不用改代码，从 engine 自动读。

**2. 归一化参数** —— 命令行给就行：

```bash
# 用 0.5/0.5/0.5 归一化的模型
./build/trt_infer --engine model.engine --image test.jpg \
    --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5
```

**3. 后处理** —— 如果模型输出的不是分类 logits（比如检测框、分割掩膜），改 `main.cpp` 里的 `print_output()`。它现在对非分类形状只打印统计量，你可以在那里加自己的解析。

如果想引入 OpenCV DNN 之外的前处理（比如 letterbox 保持长宽比），改 `preprocess()` 即可。

## 常见问题

**编译报 `NvInfer.h: No such file or directory`**

头文件在 `/usr/include/aarch64-linux-gnu/`，不在 `/usr/include` 根下。CMake 里的 `TRT_INC_HINTS` 已包含该路径，如果还找不到，手动指定：

```bash
cmake .. -DNVINFER_INCLUDE_DIR=/usr/include/aarch64-linux-gnu
```

**`fatal error: filesystem: 没有那个文件或目录`**

gcc 7.5 没有 C++17 的 `<filesystem>`。本仓库代码未使用该头文件（目录遍历用 POSIX `dirent.h`）；若你的工程需要，改用 `<experimental/filesystem>` 并链接 `stdc++fs`。

**建引擎卡住不动**

先用 `watch -n 5 'ps -o pid,etime,time,%cpu,stat -C trtexec'` 看 CPU 时间涨不涨。不涨就是死锁，把 `--workspace` 从 1024 降到 512 重试。

**推理报 `目前只支持 4 维输入`**

工具只支持 NCHW 四维输入。如果模型输入是动态 shape，建引擎时要固定尺寸（`trtexec --shapes=input:1x3x640x640`）。

更多问题见 `TX2从零部署.md` 的「五个必看的坑」与「报错对照」两节。

## 说明

示例模型用 torchvision 的 ResNet18，权重首次运行自动下载。仓库不含任何预训练模型或引擎文件——**引擎必须在目标设备上构建**，跨设备拷贝会因 TensorRT 版本或 GPU 架构不同而失效。

---

> 本文档与示例代码基于实际部署过程整理，环境版本、模型结构、命令参数因项目而异，**具体操作请结合自己的实际情况调整，仅供参考**。
