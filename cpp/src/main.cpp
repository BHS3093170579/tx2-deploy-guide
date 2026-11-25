// 最小可用的 TensorRT 推理示例：ResNet18 图像分类
//
// 用法：
//   ./trt_demo --engine resnet18.engine --image cat.jpg
//
// 流程：读图 → resize → 归一化 → CHW → 推理 → softmax → 打印 top-5

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "trt_engine.h"

namespace {

constexpr int kSize = 224;
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

// BGR uint8 → RGB 归一化 CHW float
bool preprocess(const cv::Mat& bgr, std::vector<float>& out) {
    if (bgr.empty()) return false;
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(kSize, kSize), 0, 0, cv::INTER_AREA);

    out.assign(3 * kSize * kSize, 0.0f);
    for (int y = 0; y < kSize; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kSize; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float v = row[x][c] / 255.0f;
                out[c * kSize * kSize + y * kSize + x] = (v - kMean[c]) / kStd[c];
            }
        }
    }
    return true;
}

std::vector<int> top_k(const std::vector<float>& logits, int k) {
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int n = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(n);
    return idx;
}

float softmax_at(const std::vector<float>& logits, int i) {
    const float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float v : logits) sum += std::exp(static_cast<double>(v - mx));
    return static_cast<float>(std::exp(static_cast<double>(logits[i] - mx)) / sum);
}

}  // namespace

int main(int argc, char** argv) {
    std::string engine_path, image_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
        else if (a == "--image" && i + 1 < argc) image_path = argv[++i];
    }
    if (engine_path.empty() || image_path.empty()) {
        std::printf("用法: %s --engine model.engine --image test.jpg\n", argv[0]);
        return 1;
    }

    TrtEngine engine;
    if (!engine.load(engine_path)) {
        std::fprintf(stderr, "加载失败: %s\n", engine.last_error().c_str());
        return 1;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "读不到图片: %s\n", image_path.c_str());
        return 1;
    }

    std::vector<float> input;
    if (!preprocess(img, input)) {
        std::fprintf(stderr, "预处理失败\n");
        return 1;
    }

    std::vector<float> output;
    // 首张较慢（cuDNN 选算法），先跑一次预热
    if (!engine.infer(input, output)) {
        std::fprintf(stderr, "推理失败: %s\n", engine.last_error().c_str());
        return 1;
    }
    std::printf("[warmup] %.1f ms\n", engine.last_ms());

    const int rounds = 5;
    double total = 0.0;
    for (int i = 0; i < rounds; ++i) {
        if (!engine.infer(input, output)) {
            std::fprintf(stderr, "推理失败\n");
            return 1;
        }
        total += engine.last_ms();
    }
    std::printf("[perf] 稳态 %.1f ms/张 (%.1f fps)\n", total / rounds, 1000.0 * rounds / total);

    std::printf("\nTop-5:\n");
    for (int idx : top_k(output, 5)) {
        std::printf("  class %-4d  prob %.4f\n", idx, softmax_at(output, idx));
    }
    return 0;
}
