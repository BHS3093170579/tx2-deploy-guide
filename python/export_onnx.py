"""导出一个 ResNet18 分类模型到 ONNX，用于验证部署链路。

用法：
    pip install torch torchvision onnx
    python export_onnx.py --out resnet18.onnx
"""

import argparse

import torch
import torchvision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resnet18.onnx")
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,      # 老版本 TensorRT 对 dynamo 导出的图兼容差
    )

    # 立刻用 PyTorch 跑一遍，留作后续 ONNX 比对的基准
    with torch.no_grad():
        ref = model(dummy)
    top5 = torch.topk(ref, 5, dim=1)
    print("导出完成:", args.out)
    print("PyTorch 参考 top-5:", top5.indices.tolist()[0])
    print("PyTorch 参考 top-5 概率:", [round(float(v), 4) for v in top5.values[0]])


if __name__ == "__main__":
    main()
