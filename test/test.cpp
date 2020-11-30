#include <iostream>
#include "../include/trt_east_base.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main()
{
    std::shared_ptr<TRTEastBase> east = CreateEast();

    cv::Mat img;
    std::string ini_path = "../config/trt_east.ini";
    std::string onnx_path = "../models/east.onnx";
    std::string engine_path = "../models/east.engine";
    std::string img_path = "../models/test.jpg";
    img = cv::imread(img_path);
    east->init(ini_path);
    east->load_model(onnx_path, engine_path);
    std::vector<TRTEastResult> results;
    //阈值默认应该设置为0.5
    results = east->detect(img, 0.5);
    for (auto result : results) {
        // std::cout << result.label << std::endl;
        std::cout << result.score << std::endl;
        for (int i = 0; i < 8; i++) {
            std::cout << result.box_coordinates[i] << " ";
        }
        std::cout << "" << std::endl;
    }

    return 0;
}
