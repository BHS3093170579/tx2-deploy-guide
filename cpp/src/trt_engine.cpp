#include "trt_engine.h"

#include <cstdio>
#include <fstream>
#include <numeric>

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if (sev == Severity::kERROR || sev == Severity::kINTERNAL_ERROR) {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

Logger g_logger;

bool cuda_ok(const char* what, cudaError_t code) {
    if (code == cudaSuccess) return true;
    std::fprintf(stderr, "[CUDA] %s failed: %s\n", what, cudaGetErrorString(code));
    return false;
}

}  // namespace

TrtEngine::~TrtEngine() { release(); }

void TrtEngine::release() {
    if (d_in_) cudaFree(d_in_);
    if (d_out_) cudaFree(d_out_);
    if (stream_) cudaStreamDestroy(stream_);
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_end_) cudaEventDestroy(ev_end_);
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
    d_in_ = d_out_ = nullptr;
    stream_ = nullptr;
    ev_start_ = ev_end_ = nullptr;
    context_ = nullptr;
    engine_ = nullptr;
    runtime_ = nullptr;
}

int TrtEngine::input_numel() const { return in_numel_; }
int TrtEngine::output_numel() const { return out_numel_; }

bool TrtEngine::load(const std::string& engine_path) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) {
        err_ = "无法打开引擎文件: " + engine_path;
        return false;
    }
    f.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    f.read(blob.data(), static_cast<std::streamsize>(size));
    f.close();

    runtime_ = nvinfer1::createInferRuntime(g_logger);
    if (!runtime_) {
        err_ = "createInferRuntime 失败";
        return false;
    }
    engine_ = runtime_->deserializeCudaEngine(blob.data(), size, nullptr);
    if (!engine_) {
        err_ = "反序列化失败（引擎与当前 TensorRT 版本可能不兼容）";
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (!context_) {
        err_ = "createExecutionContext 失败";
        return false;
    }

    // 按名字定位 binding：比按索引可靠，不怕导出时顺序变化
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        const char* name = engine_->getBindingName(i);
        if (engine_->bindingIsInput(i)) {
            in_index_ = i;
            in_name_ = name;
        } else {
            out_index_ = i;
            out_name_ = name;
        }
    }
    if (in_index_ < 0 || out_index_ < 0) {
        err_ = "找不到输入或输出 binding";
        return false;
    }

    auto copy_dims = [](const nvinfer1::Dims& d, std::vector<int64_t>& out) {
        out.clear();
        for (int i = 0; i < d.nbDims; ++i) out.push_back(d.d[i]);
    };
    copy_dims(engine_->getBindingDimensions(in_index_), in_dims_);
    copy_dims(engine_->getBindingDimensions(out_index_), out_dims_);

    in_numel_ = 1;
    for (auto v : in_dims_) in_numel_ *= static_cast<int>(v);
    out_numel_ = 1;
    for (auto v : out_dims_) out_numel_ *= static_cast<int>(v);

    if (!init_buffers()) return false;

    std::printf("[TRT] engine=%s\n", engine_path.c_str());
    std::printf("[TRT] input  '%s' dims=[", in_name_.c_str());
    for (size_t i = 0; i < in_dims_.size(); ++i)
        std::printf("%s%lld", i ? "," : "", static_cast<long long>(in_dims_[i]));
    std::printf("]\n[TRT] output '%s' dims=[", out_name_.c_str());
    for (size_t i = 0; i < out_dims_.size(); ++i)
        std::printf("%s%lld", i ? "," : "", static_cast<long long>(out_dims_[i]));
    std::printf("]\n");
    return true;
}

bool TrtEngine::init_buffers() {
    if (!cuda_ok("cudaMalloc(d_in)", cudaMalloc(reinterpret_cast<void**>(&d_in_),
                                                in_numel_ * sizeof(float))))
        return false;
    if (!cuda_ok("cudaMalloc(d_out)", cudaMalloc(reinterpret_cast<void**>(&d_out_),
                                                 out_numel_ * sizeof(float))))
        return false;
    if (!cuda_ok("cudaStreamCreate", cudaStreamCreate(&stream_))) return false;
    if (!cuda_ok("cudaEventCreate", cudaEventCreate(&ev_start_))) return false;
    if (!cuda_ok("cudaEventCreate", cudaEventCreate(&ev_end_))) return false;
    return true;
}

bool TrtEngine::infer(const std::vector<float>& input, std::vector<float>& output,
                      bool copy_back) {
    if (static_cast<int>(input.size()) != in_numel_) {
        err_ = "输入尺寸不匹配";
        return false;
    }

    if (!cuda_ok("cudaMemcpyAsync(H2D)",
                 cudaMemcpyAsync(d_in_, input.data(), in_numel_ * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_)))
        return false;

    // TensorRT 8.x：enqueueV2 接受 void* 数组，按下标对应 binding
    // 用 getNbBindings() 分配，多输出的模型也不会越界
    std::vector<void*> bindings(static_cast<size_t>(engine_->getNbBindings()), nullptr);
    bindings[static_cast<size_t>(in_index_)] = d_in_;
    bindings[static_cast<size_t>(out_index_)] = d_out_;

    cudaEventRecord(ev_start_, stream_);
    const bool ok = context_->enqueueV2(bindings.data(), stream_, nullptr);
    cudaEventRecord(ev_end_, stream_);
    cudaEventSynchronize(ev_end_);
    cudaEventElapsedTime(&last_ms_, ev_start_, ev_end_);

    if (!ok) {
        err_ = "enqueueV2 执行失败";
        return false;
    }

    if (copy_back) {
        output.resize(out_numel_);
        if (!cuda_ok("cudaMemcpyAsync(D2H)",
                     cudaMemcpyAsync(output.data(), d_out_, out_numel_ * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_)))
            return false;
        cudaStreamSynchronize(stream_);
    }
    return true;
}
