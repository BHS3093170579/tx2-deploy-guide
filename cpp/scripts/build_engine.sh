#!/usr/bin/env bash
# 把 ONNX 编译成 TensorRT 引擎
#
# 用法:
#   ./scripts/build_engine.sh model.onnx                 # FP16（推荐）
#   ./scripts/build_engine.sh model.onnx fp32            # FP32
#   ./scripts/build_engine.sh model.onnx fp16 512        # 指定 workspace(MB)
#
# 说明:
#   * workspace 越大，可选的高性能 kernel 越多，但构建越慢、也可能卡死
#   * Jetson 上建议先用 512，构建成功后再按需加大
#   * 构建在 TX2 上通常需要几分钟到十几分钟，属正常

set -euo pipefail

ONNX="${1:-}"
PRECISION="${2:-fp16}"
WORKSPACE="${3:-512}"

if [ -z "$ONNX" ]; then
    echo "用法: $0 <model.onnx> [fp16|fp32] [workspace_MB]"
    exit 1
fi
if [ ! -f "$ONNX" ]; then
    echo "找不到文件: $ONNX"
    exit 1
fi

ENGINE="${ONNX%.onnx}.${PRECISION}.engine"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [ ! -x "$TRTEXEC" ]; then
    echo "找不到 trtexec: $TRTEXEC"
    echo "可以通过环境变量指定: TRTEXEC=/path/to/trtexec $0 ..."
    exit 1
fi

EXTRA=""
if [ "$PRECISION" = "fp16" ]; then
    EXTRA="--fp16"
fi

echo "==> 输入   : $ONNX"
echo "==> 精度   : $PRECISION"
echo "==> 工作区 : ${WORKSPACE} MB"
echo "==> 输出   : $ENGINE"
echo

# --buildOnly 只构建不跑基准，省时间
"$TRTEXEC" \
    --onnx="$ONNX" \
    $EXTRA \
    --workspace="$WORKSPACE" \
    --buildOnly \
    --saveEngine="$ENGINE" 2>&1 | tail -30

if [ -f "$ENGINE" ]; then
    echo
    echo "构建成功: $ENGINE ($(du -h "$ENGINE" | cut -f1))"
    echo "试跑: ./build/trt_infer --engine $ENGINE --image 你的图.jpg"
else
    echo "构建失败，没有生成 engine 文件"
    exit 1
fi
