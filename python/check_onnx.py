"""用 onnxruntime 验证导出的 ONNX 模型，并给出 TensorRT 构建前的自检项。

用法：
    pip install onnx onnxruntime numpy
    python check_onnx.py --model resnet18.onnx
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18.onnx")
    args = ap.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort

    print("=" * 50)
    print("1. 模型结构检查")
    print("=" * 50)
    m = onnx.load(args.model)
    onnx.checker.check_model(m)      # 结构不合法会在这里抛异常
    opset = m.opset_import[0].version if m.opset_import else -1
    print(f"  IR 版本   : {m.ir_version}")
    print(f"  opset     : {opset}")
    print(f"  节点数    : {len(m.graph.node)}")
    print(f"  算子种类  : {len({n.op_type for n in m.graph.node})}")

    # TensorRT 8.2 已知不支持的写法：TopK 的 K 必须是常量
    bad = [n.name for n in m.graph.node if n.op_type == "TopK"]
    if bad:
        print(f"  [警告] 存在 TopK 节点 {len(bad)} 个，若 K 来自动态输入，TRT 8.2 会报错")

    print()
    print("=" * 50)
    print("2. 数值验证（随机输入）")
    print("=" * 50)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"  输入: {inp.name} {inp.shape} {inp.type}")
    print(f"  输出: {out.name} {out.shape} {out.type}")

    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.randn(*shape).astype("float32")
    y = sess.run(None, {inp.name: x})[0]
    flat = y.reshape(-1)

    # softmax 后看 top-5
    e = np.exp(flat - flat.max())
    p = e / e.sum()
    top5 = np.argsort(-p)[:5]
    print("  ORT top-5:", top5.tolist())
    print("  ORT top-5 概率:", [round(float(p[i]), 4) for i in top5])

    print()
    print("=" * 50)
    print("3. 下一步")
    print("=" * 50)
    print(f"  trtexec --onnx={args.model} --fp16 --workspace=512 --saveEngine=model.fp16.engine")
    print("  然后用 cpp/ 里的 trt_demo 加载引擎对比 top-5 是否一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
