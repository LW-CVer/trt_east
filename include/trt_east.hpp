#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include "../include/util.hpp"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "opencv2/core/mat.hpp"
#include "trt_east_base.hpp"
using namespace nvinfer1;

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

class Logger_east : public nvinfer1::ILogger
{
   public:
    Logger_east(Severity severity = Severity::kINFO)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the
        // reportable
        if (severity > reportableSeverity)
            return;

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

class TRTEast : public TRTEastBase
{
   public:
    TRTEast();
    ~TRTEast() override;
    int init(const std::string& ini_path);
    int load_model(std::string& onnx_file, std::string& engine_file);
    std::vector<TRTEastResult> detect(cv::Mat& image, double score_threshold);

   private:
    nvinfer1::IExecutionContext* m_context;
    nvinfer1::IRuntime* m_runtime;
    nvinfer1::ICudaEngine* m_engine;
    cudaStream_t m_stream;

    std::vector<void*> m_buffers;  //一个存输入两个存输出 GPU
    float* m_results[2];           //接受trt输出结果
    std::vector<TRTEastResult> m_detection_results;
    int m_gpu_index;
    std::vector<float> m_ratios;
    int m_feature_height;
    int m_feature_width;
    float m_box_threshold;
    float m_nms_threshold;
    void doInference(cv::Mat& img);
    void detect_initialise();
    std::vector<std::vector<float>> process(double score_threshold,
                                            std::vector<float>& final_scores,
                                            std::vector<float>& avg_scores);
    bool CheckBox(std::vector<float>& box, float final_score, float avg_score,
                  double score_threshold);
};
