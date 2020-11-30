#ifndef __TRT_EAST_UTIL_HPP__
#define __TRT_EAST_UTIL_HPP__

#include <opencv2/core/mat.hpp>
#include <vector>

namespace trt_east {

std::vector<float> ResizeImage(cv::Mat& image, cv::Mat& resized_img,
                               int max_side_len = 2400);

std::vector<std::vector<float>> RestoreRectangle(
    std::vector<std::pair<int, int>>& yx_indexs,
    std::vector<std::vector<float>>& coords, std::vector<float>& angles,
    std::vector<float>& scores);

void GetScore(std::vector<std::vector<float>>& boxes, float* f_score,
              std::vector<float>& final_scores, std::vector<float>& avg_scores,
              int width);

int GetDist(int x1, int y1, int x2, int y2);
}  // namespace trt_east
#endif  //__TF_EAST_UTIL_HPP__
