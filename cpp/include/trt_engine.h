#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

// 一个最小的 TensorRT 推理封装：
//   load()  —— 反序列化引擎、建执行上下文、按名字绑定输入输出
//   infer() —— 上传输入、执行、取回输出，并记录耗时
class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // 从 .engine 文件加载，成功返回 true
    bool load(const std::string& engine_path);

    // 单次推理：input 为预处理后的 CHW 浮点数据
    bool infer(const std::vector<float>& input, std::vector<float>& output,
               bool copy_back = true);

    // 最近一次推理的 GPU 耗时（毫秒）
    double last_ms() const { return last_ms_; }

    // 输入/输出张量形状，例如 {1, 3, 640, 640}
    const std::vector<int64_t>& input_dims() const { return in_dims_; }
    const std::vector<int64_t>& output_dims() const { return out_dims_; }

    int input_numel() const;
    int output_numel() const;

    // 输出缓冲的显存指针，配合 infer(..., copy_back=false) 可省去回拷
    float* output_device() const { return d_out_; }

    const std::string& last_error() const { return err_; }

private:
    bool init_buffers();
    void release();

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    int in_index_ = -1;
    int out_index_ = -1;
    std::string in_name_;
    std::string out_name_;
    std::vector<int64_t> in_dims_;
    std::vector<int64_t> out_dims_;

    float* d_in_ = nullptr;
    float* d_out_ = nullptr;
    int in_numel_ = 0;
    int out_numel_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_end_ = nullptr;

    double last_ms_ = 0.0;
    std::string err_;
};
