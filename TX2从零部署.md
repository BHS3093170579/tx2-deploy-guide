# TX2 从零部署全流程

> 在 Jetson TX2 上从零编译一个 C++ + TensorRT 推理工程，跑通并取回结果。
> 全程约 25 分钟（建引擎占大头）。

---

## 总览

| 阶段 | 在哪执行 | 干什么 | 产物 | 耗时 |
|---|---|---|---|---|
| 0 | Windows | 准备工程源码 | 本地 C++ 工程 | 1 min |
| 1 | 板子 | 查环境 | — | 10 s |
| 2 | Windows | 准备 ONNX 模型 | `model.onnx` | 视模型而定 |
| 3 | Windows | 传文件上去 | 板子上有源码和模型 | 1 min |
| 4 | 板子 | 编译 | 可执行文件 | 40 s |
| 5 | 板子 | 建 TRT 引擎 | `model.fp16.engine` | 3–17 min |
| 6 | 板子 | 跑推理 | 结果文件 | 1 s |
| 7 | Windows | 取回结果 | 本地文件 | 5 s |

---

## 阶段 0　准备工程源码

工程目录结构必须是这样（`CMakeLists.txt` 在 `app/` 下）：

```
app/
├── CMakeLists.txt
├── include/
│   ├── trt_engine.h
│   ├── postprocess.h
│   └── ...
└── src/
    ├── main.cpp
    ├── trt_engine.cpp
    └── ...
```

如果工程在一台机器上、要搬到另一台，用 scp 整个目录：

```powershell
scp -r "D:\你的工程\app" tx2@ubuntu.local:/home/tx2/project/
```

---

## 阶段 1　板子环境检查

```powershell
ssh tx2@ubuntu.local
```

进去后逐项确认：

```bash
echo "=== 系统 ==="      ; head -1 /etc/nv_tegra_release
echo "=== CUDA ==="      ; nvcc --version | grep release
echo "=== TensorRT ==="  ; ls /usr/include/aarch64-linux-gnu/NvInfer.h
echo "=== OpenCV ==="    ; pkg-config --modversion opencv4
echo "=== 编译器 ==="    ; g++ --version | head -1
echo "=== 磁盘 ==="      ; df -h / | tail -1
```

**必须满足**：

- TensorRT 头文件存在（没有就装 JetPack 对应组件）
- OpenCV ≥ 4.0
- 磁盘剩余 **≥ 3 GB**

典型 TX2 环境（JetPack 4.6.6）：

| 项目 | 版本 |
|---|---|
| 系统 | L4T R32.7.6 |
| CUDA | 10.2 |
| TensorRT | 8.2.1 |
| OpenCV | 4.1.1 |
| g++ | 7.5.0 |

---

## 阶段 2　准备 ONNX 模型

在 PC 上把训练好的模型导出成 ONNX，固定输入尺寸（动态 shape 在 TX2 上会增加复杂度）：

```python
import torch

torch.onnx.export(
    model,
    torch.randn(1, 3, 640, 640),
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamo=False,          # 老版本 TRT 对 dynamo 导出的图兼容差
)
```

**两个注意点**：

1. **opset 别太高**。TRT 8.2 对 opset 17+ 的部分算子支持不全，用 11 最稳
2. **验证一次**。用 onnxruntime 跑一遍，和 PyTorch 输出对比，差在 1e-2 以内才算正常

如果模型还有额外的权重文件（特征库、embedding、校准表等），一并准备好，后面要一起传上去。

---

## 阶段 3　传文件到板子

```powershell
# 1. 源码
scp -r "D:\你的工程\app\src"      tx2@ubuntu.local:/home/tx2/project/app/
scp -r "D:\你的工程\app\include"  tx2@ubuntu.local:/home/tx2/project/app/
scp    "D:\你的工程\app\CMakeLists.txt" tx2@ubuntu.local:/home/tx2/project/app/

# 2. ONNX 模型
scp "D:\模型\model.onnx" tx2@ubuntu.local:/home/tx2/project/models/

# 3. 额外权重（如果有）
scp -r "D:\模型\weights" tx2@ubuntu.local:/home/tx2/project/models/
```

传完在板子上验证：

```bash
ls -lh ~/project/app/src/ ~/project/models/
```

---

## 阶段 4　编译（板子上）

```bash
cd ~/project/app
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

**期望最后一行**：

```
[100%] Built target 你的目标名
```

`-j4` 是因为 TX2 只有 6 核且内存有限，开满容易 OOM。

**编译报错怎么定位**：

```bash
make -j4 2>&1 | grep -E "error" | head -10
```

只看第一条 error，后面的往往是连锁反应。

---

## 阶段 5　建 TensorRT 引擎（板子上）

```bash
cd ~/project/models

/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --fp16 \
  --workspace=512 \
  --saveEngine=model.fp16.engine
```

**耗时**：视模型大小，从 3 分钟到 20 分钟不等。

**成功标志**：

```
[TRT] Engine built in 170.694 sec.
&&&& PASSED TensorRT.trtexec [TensorRT v8201]
```

### 怎么判断是慢还是卡死

另开一个 SSH 窗口，跑两条监控：

```bash
# 看 CPU 时间涨不涨
watch -n 5 'ps -o pid,etime,time,%cpu,stat -C trtexec'

# 看 GPU 有没有在动（TX2 没有 nvidia-smi，用 sysfs）
while true; do echo "GPU: $(cat /sys/devices/gpu.0/load)/1000"; sleep 2; done
```

判断标准：

| 现象 | 结论 |
|---|---|
| CPU 时间在涨 + GPU 有跳动 | 正常，继续等 |
| CPU 时间不动 + GPU 恒为 0 | 死锁，Ctrl+C 重来 |
| CPU 只占 10–15% | 正常，在等 GPU 干活 |

> `--workspace` 不是越大越好。TX2 上设 1024 反而容易卡死，**512 最稳**。

---

## 阶段 6　跑推理（板子上）

```bash
cd ~/project/app/build

./你的可执行文件 \
  --engine ~/project/models/model.fp16.engine \
  --image ~/project/data/test.png
```

**期望输出**（示例）：

```
[TRT] binding 0 input  <- dims=[1,3,640,640]
[TRT] binding 1 output -> dims=[1,1536,80,80]
load_ms=4300

test.png   total=218.6 ms
```

三个要核对的数：

| 项目 | 说明 |
|---|---|
| 首张耗时 | 明显高于后续，属预热，正常 |
| 稳态耗时 | 才是真实性能 |
| 输出数值 | 和 PC 上跑的结果对比，差异应在 1% 以内 |

如果程序带 `--repeat N` 这类参数，用它跑多次、跳过首张统计稳态：

```bash
./你的可执行文件 --engine ... --image ... --repeat 5
```

> 注意：有些实现的 `--repeat` 模式为了不拖慢测速会跳过写文件。要出图就单独跑一次，别带 `--repeat`。

---

## 阶段 7　取回结果

```powershell
scp "tx2@ubuntu.local:/home/tx2/project/输出文件" "D:\本地目录\输出文件"
```

传完立刻验证，别看命令没报错就当成功了：

```powershell
Get-ChildItem "D:\本地目录"
```

---

## 五个必看的坑

### 1. 换 WiFi 后板子 IP 会变

板子走 DHCP，换网络后地址会重新分配。用主机名代替 IP 一劳永逸：

```powershell
ssh tx2@ubuntu.local
```

解析不了时，两条命令定位：

```powershell
Resolve-DnsName ubuntu                         # 当前 IP
arp -a | Select-String "10\."                  # 局域网设备，MAC 以 00:04:4b 开头的是 NVIDIA
```

### 2. scp 别用 `~`

PowerShell 对 `~` 的处理和 Linux 不同，可能传成奇怪路径。一律用绝对路径：

```powershell
# 不推荐
scp -r tx2@ubuntu.local:~/project "D:\..."

# 推荐
scp -r tx2@ubuntu.local:/home/tx2/project "D:\..."
```

### 3. scp 成功不等于传到了

`scp -r 源目录 目标目录\` 会在目标下**再套一层同名目录**，看起来空白其实是嵌套了。

单文件传输时把目标写成完整文件路径：

```powershell
scp "tx2@ubuntu.local:/home/tx2/project/out.jpg" "D:\本地\out.jpg"
```

### 4. 文件时间戳能判断有没有真的生成

程序跑完没报错，不代表产物是新的。看一眼时间：

```bash
ls -lh ~/project/output/*
```

时间是几天前的，说明这次根本没写盘。

### 5. 磁盘随时会满

TX2 的 eMMC 通常只有 28 GB，JetPack 系统占掉 20 GB 以上。

```bash
du -sh ~/* 2>/dev/null | sort -rh | head -10
df -h /
```

常见可清理的：

```bash
rm -rf ~/.cache/*
sudo apt clean
```

**不要随手删 conda 环境或别人的工程目录**，先确认哪些还在用。

---

## 附：报错对照

| 报错 | 原因 | 解决 |
|---|---|---|
| `fatal error: filesystem: 没有那个文件或目录` | gcc 7.5 没有 C++17 `<filesystem>` | 用 `<experimental/filesystem>`，链接 `stdc++fs` |
| `'fs' has not been declared` | `__has_include` 探测在 gcc 7.5 失效 | 直接硬编码 `std::experimental::filesystem` |
| `has no member named 'is_regular_file'` | experimental 版 API 不同 | 改 `fs::is_regular_file(path, ec)` |
| `setMemoryPoolLimit` 未声明 | 那是 TRT 8.5+ 的 API | 改 `setMaxWorkspaceSize(ws)` |
| `invalid conversion from 'int' to 'cudaError_t'` | C++ 严格类型检查 | `static_cast<cudaError_t>(code)` |
| `TopKLayer: k <= MAX_TOPK_K` | TRT 8.2 的 TopK 上限是 1024 | 改图去掉大 K 的 TopK，或降 K |
| `importConv: bias tensor is required to be an initializer` | 导出时 bias 被包了 Identity | 用 onnx-simplifier 折叠 Identity |
| `Unable to load engine` | 引擎与当前 TRT 版本不兼容 | 在目标机上重新建引擎 |
| 推理比预期慢 2 倍 | 首张预热 / 没开 FP16 | 用 `--repeat` 看稳态，确认 `--fp16` |
| 显存不足 | 其它进程占用 | `sudo tegrastats` 查看，关掉占用进程 |

---

## 一句话总结

**PC 负责导出模型，板子负责编译和推理，文件用 scp 来回传。**

真正需要注意的只有三处：`--workspace=512`、opset 别太高、首张会慢。

---

> 本文档基于实际部署过程整理，环境版本、模型结构、命令参数因项目而异，**具体操作请结合自己的实际情况调整，仅供参考**。
