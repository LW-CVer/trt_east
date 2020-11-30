#ifndef __TRT_EAST_BASE_HPP__
#define __TRT_EAST_BASE_HPP__
#include <memory>
#include <string>
#include <vector>
#include "opencv2/core/mat.hpp"
struct TRTEastResult
{
    int label;
    int box_coordinates[8];
    float score;
};

class TRTEastBase
{
   public:
    virtual ~TRTEastBase() = default;
    virtual int init(const std::string& ini_path) = 0;
    virtual int load_model(std::string& onnx_file,
                           std::string& engine_file) = 0;
    virtual std::vector<TRTEastResult> detect(cv::Mat& image,
                                              double score_threshold) = 0;
};

std::shared_ptr<TRTEastBase> CreateEast();
#endif
