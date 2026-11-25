// 通用 TensorRT 推理工具（不绑定具体模型）
//
// 用法：
//   ./trt_infer --engine model.engine --image test.jpg
//   ./trt_infer --engine model.engine --dir ./images
//   ./trt_infer --engine model.engine --image test.jpg --repeat 10
//   ./trt_infer --engine model.engine --image test.jpg --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5
//
// 设计要点：
//   * 输入尺寸、通道数从 engine 里读，不写死 224
//   * 归一化参数可从命令行覆盖，默认 ImageNet
//   * 输出形状未知时也能跑：分类输出打印 top-5，其它输出打印形状与统计量

#include <dirent.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "trt_engine.h"

namespace {

struct Options {
    std::string engine;
    std::string image;
    std::string dir;
    int repeat = 5;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

// 解析 "0.485,0.456,0.406" 这样的三通道参数
bool parse3(const char* s, float out[3]) {
    float a = 0, b = 0, c = 0;
    if (std::sscanf(s, "%f,%f,%f", &a, &b, &c) != 3) return false;
    out[0] = a;
    out[1] = b;
    out[2] = c;
    return true;
}

bool is_image(const std::string& name) {
    const char* exts[] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const char* e : exts) {
        const size_t n = std::strlen(e);
        if (lower.size() > n && lower.compare(lower.size() - n, n, e) == 0) return true;
    }
    return false;
}

std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent* ent = readdir(d)) {
        const std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        if (is_image(name)) out.push_back(dir + "/" + name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// BGR uint8 → RGB 归一化 CHW float；尺寸取自 engine
bool preprocess(const cv::Mat& bgr, int w, int h, const float mean[3], const float std[3],
                std::vector<float>& out) {
    if (bgr.empty()) return false;
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    // 缩小用 INTER_AREA 更稳，放大用 INTER_LINEAR
    const int mode = (bgr.cols > w || bgr.rows > h) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(rgb, resized, cv::Size(w, h), 0, 0, mode);

    out.assign(3 * w * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float v = row[x][c] / 255.0f;
                out[c * w * h + y * w + x] = (v - mean[c]) / std[c];
            }
        }
    }
    return true;
}

void print_output(const std::vector<float>& data, const std::vector<int64_t>& dims) {
    int64_t numel = 1;
    for (int64_t d : dims) numel *= d;
    std::printf("  输出形状: [");
    for (size_t i = 0; i < dims.size(); ++i) std::printf("%s%lld", i ? ", " : "", (long long)dims[i]);
    std::printf("]  共 %lld 个值\n", (long long)numel);

    if (data.empty()) return;

    // 形状像 [1, N]（N 不太大）就当作分类 logits，打印 top-5
    const bool looks_like_logits =
        (dims.size() == 2 && dims[0] == 1 && dims[1] <= 10000) ||
        (dims.size() == 1 && dims[0] <= 10000);
    if (looks_like_logits) {
        const int n = std::min<int>(5, static_cast<int>(data.size()));
        std::vector<int> idx(data.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                          [&](int a, int b) { return data[a] > data[b]; });
        const float mx = *std::max_element(data.begin(), data.end());
        double sum = 0.0;
        for (float v : data) sum += std::exp(static_cast<double>(v - mx));
        std::printf("  Top-%d:\n", n);
        for (int i = 0; i < n; ++i) {
            const float p = static_cast<float>(std::exp(static_cast<double>(data[idx[i]] - mx)) / sum);
            std::printf("    class %-5d  prob %.4f\n", idx[i], p);
        }
        return;
    }

    // 其它形状：给统计量和前几个值，方便确认数值是否合理
    float mn = data[0], mxs = data[0];
    double sm = 0.0;
    for (float v : data) {
        mn = std::min(mn, v);
        mxs = std::max(mxs, v);
        sm += v;
    }
    std::printf("  统计: min %.4f  max %.4f  mean %.4f\n", mn, mxs, sm / data.size());
    std::printf("  前 8 个: ");
    for (int i = 0; i < std::min<int>(8, static_cast<int>(data.size())); ++i) {
        std::printf("%.4f ", data[i]);
    }
    std::printf("\n");
}

void usage(const char* prog) {
    std::printf(
        "用法: %s --engine <model.engine> (--image <file> | --dir <folder>) [选项]\n\n"
        "选项:\n"
        "  --repeat N          稳态重复次数，默认 5（首次不计入）\n"
        "  --mean r,g,b        归一化均值，默认 ImageNet 0.485,0.456,0.406\n"
        "  --std  r,g,b        归一化标准差，默认 ImageNet 0.229,0.224,0.225\n"
        "\n示例:\n"
        "  %s --engine resnet18.engine --image cat.jpg\n"
        "  %s --engine model.engine --dir ./images --repeat 10\n",
        prog, prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--engine" && i + 1 < argc) opt.engine = argv[++i];
        else if (a == "--image" && i + 1 < argc) opt.image = argv[++i];
        else if (a == "--dir" && i + 1 < argc) opt.dir = argv[++i];
        else if (a == "--repeat" && i + 1 < argc) opt.repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--mean" && i + 1 < argc) {
            if (!parse3(argv[++i], opt.mean)) { std::fprintf(stderr, "--mean 需要三个逗号分隔的数\n"); return 1; }
        } else if (a == "--std" && i + 1 < argc) {
            if (!parse3(argv[++i], opt.std)) { std::fprintf(stderr, "--std 需要三个逗号分隔的数\n"); return 1; }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        }
    }
    if (opt.engine.empty() || (opt.image.empty() && opt.dir.empty())) {
        usage(argv[0]);
        return 1;
    }

    TrtEngine engine;
    if (!engine.load(opt.engine)) {
        std::fprintf(stderr, "引擎加载失败: %s\n", engine.last_error().c_str());
        return 1;
    }

    const auto& in = engine.input_dims();
    if (in.size() != 4) {
        std::fprintf(stderr, "目前只支持 4 维输入（NCHW），实际是 %zu 维\n", in.size());
        return 1;
    }
    const int channels = static_cast<int>(in[1]);
    const int height = static_cast<int>(in[2]);
    const int width = static_cast<int>(in[3]);
    if (channels != 3) {
        std::fprintf(stderr, "目前只支持 3 通道输入，engine 要求 %d 通道\n", channels);
        return 1;
    }
    std::printf("引擎已加载: 输入 %dx%dx%d\n", channels, height, width);

    std::vector<std::string> files;
    if (!opt.image.empty()) files.push_back(opt.image);
    else {
        files = list_images(opt.dir);
        if (files.empty()) {
            std::fprintf(stderr, "目录里没有图片: %s\n", opt.dir.c_str());
            return 1;
        }
    }

    std::vector<float> input, output;
    double sum_ms = 0.0;
    int n_ok = 0;

    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            std::fprintf(stderr, "[跳过] 读不到图片: %s\n", files[i].c_str());
            continue;
        }
        if (!preprocess(img, width, height, opt.mean, opt.std, input)) {
            std::fprintf(stderr, "[跳过] 预处理失败: %s\n", files[i].c_str());
            continue;
        }

        // 预热（cuDNN 首次会选算法，明显偏慢）
        if (!engine.infer(input, output)) {
            std::fprintf(stderr, "推理失败: %s\n", engine.last_error().c_str());
            return 1;
        }
        const double warmup = engine.last_ms();

        double total = 0.0;
        for (int r = 0; r < opt.repeat; ++r) {
            if (!engine.infer(input, output)) {
                std::fprintf(stderr, "推理失败: %s\n", engine.last_error().c_str());
                return 1;
            }
            total += engine.last_ms();
        }
        const double avg = total / opt.repeat;
        sum_ms += avg;
        ++n_ok;

        std::printf("\n[%zu/%zu] %s\n", i + 1, files.size(), files[i].c_str());
        std::printf("  耗时: 预热 %.1f ms | 稳态 %.1f ms (%.1f fps)\n", warmup, avg, 1000.0 / avg);
        print_output(output, engine.output_dims());
    }

    if (n_ok > 1) {
        std::printf("\n合计 %d 张，平均 %.1f ms/张 (%.1f fps)\n", n_ok, sum_ms / n_ok,
                    1000.0 * n_ok / sum_ms);
    }
    return 0;
}
