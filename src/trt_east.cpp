#include "../include/trt_east.hpp"
#include <fstream>
#include "../include/INIReader.hpp"
#include "../include/ini.hpp"
#include "../include/lanms.h"
#include "../include/util.hpp"
#include "opencv2/core.hpp"

TRTEast::TRTEast()
    : m_buffers{nullptr, nullptr, nullptr}, m_results{nullptr, nullptr}
{
}

TRTEast::~TRTEast()
{
    for (size_t i = 0; i < 3; ++i) {
        cudaFree(m_buffers[i]);
        // delete[] m_buffers[i];
        m_buffers[i] = nullptr;
    }
    for (size_t i = 0; i < 2; ++i) {
        delete[] m_results[i];
        m_results[i] = nullptr;
    }
}

int TRTEast::init(const std::string& ini_path)
{
    INIReader reader(ini_path);
    m_gpu_index = reader.GetInteger("device", "gpu_index", 0);
    m_box_threshold = reader.GetReal("threshold", "box_threshold", 0.1);
    m_nms_threshold = reader.GetReal("threshold", "nms_threshold", 0.2);
    return 0;
}

int TRTEast::load_model(std::string& onnx_file, std::string& engine_file)
{
    Logger_east gLogger;
    cudaSetDevice(m_gpu_index);
    std::ifstream intrt(engine_file, std::ios::binary);

    if (intrt) {
        std::cout << "load local engine..." << engine_file << std::endl;
        m_runtime = nvinfer1::createInferRuntime(gLogger);
        intrt.seekg(0, std::ios::end);
        size_t length = intrt.tellg();
        intrt.seekg(0, std::ios::beg);
        std::vector<char> data(length);
        intrt.read(data.data(), length);
        m_engine = m_runtime->deserializeCudaEngine(data.data(), length);
        std::cout << "engine loaded." << std::endl;
    } else {
        std::cout << "create engine from onnx..." << std::endl;
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            return -1;
        }

        auto network = builder->createNetworkV2(
            1U << static_cast<int>(
                NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        if (!network) {
            return -1;
        }
        auto config = builder->createBuilderConfig();
        if (!config) {
            return -1;
        }

        auto parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser) {
            return -1;
        }

        auto parsed = parser->parseFromFile(
            onnx_file.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
        if (!parsed) {
            std::cout << "parse onnx failed." << std::endl;
            return -1;
        }

        nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
        nvinfer1::Dims output_dims_1 = network->getOutput(0)->getDimensions();
        nvinfer1::Dims output_dims_2 = network->getOutput(1)->getDimensions();
        /*std::cout << input_dims.d[0] << " " << input_dims.d[1] << " "
                  << input_dims.d[2] << " " << input_dims.d[3] << std::endl;
        std::cout << output_dims_1.d[0] << " " << output_dims_1.d[1] << " "
                  << output_dims_1.d[2] << " " << output_dims_1.d[3]
                  << std::endl;
        std::cout << output_dims_2.d[0] << " " << output_dims_2.d[1] << " "
                  << output_dims_2.d[2] << " " << output_dims_2.d[3]
                  << std::endl;*/

        config->setAvgTimingIterations(1);
        config->setMinTimingIterations(1);
        config->setMaxWorkspaceSize(1 << 20);
        auto input = network->getInput(0);
        //设置动态维度
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(input->getName(),
                               nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4{1, 3, 64, 64});
        profile->setDimensions(input->getName(),
                               nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(input->getName(),
                               nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4{1, 3, 1080, 1080});
        config->addOptimizationProfile(profile);

        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(1 << 20);
        m_engine = builder->buildEngineWithConfig(*network, *config);
        nvinfer1::IHostMemory* engine_serialize = m_engine->serialize();
        std::ofstream out(engine_file.c_str(), std::ios::binary);
        out.write((char*)engine_serialize->data(), engine_serialize->size());
        std::cout << "serialize the engine to " << engine_file << std::endl;
        engine_serialize->destroy();
        parser->destroy();
        config->destroy();
        network->destroy();
        builder->destroy();
    }
    
    m_context = m_engine->createExecutionContext();
    /*
    for (int b = 0; b < m_engine->getNbBindings(); ++b) {
        if (m_engine->bindingIsInput(b))

            std::cout << "input:" << b << std::endl;
        else
            std::cout << "output:" << b << std::endl;
    }*/

    cudaStreamCreate(&m_stream);

    std::cout << "RT init done!" << std::endl;
    return 0;
}

void TRTEast::detect_initialise()
{
    for (size_t i = 0; i < 2; ++i) {
        if (m_results[i] != nullptr) {
            delete[] m_results[i];
            m_results[i] = nullptr;
        }
    }
    m_ratios.clear();
    m_detection_results.clear();
}

void TRTEast::doInference(cv::Mat& img)
{
    //要把输入img转成一维数组
    cv::Mat resized_img;
    //缩放，并转为RGB
    m_ratios = trt_east::ResizeImage(img, resized_img);
    long int input_size = resized_img.rows * resized_img.cols * 3 * 1;

    //设置engine当前的输入维度,bchw
    nvinfer1::Dims4 input_shape{1, 3, resized_img.rows, resized_img.cols};
    m_context->setBindingDimensions(0, input_shape);

    m_feature_height = resized_img.rows / 4;
    m_feature_width = resized_img.cols / 4;
    long int score_output_size =
        (resized_img.rows / 4) * (resized_img.cols / 4) * 1 * 1;
    long int geometry_output_size =
        (resized_img.rows / 4) * (resized_img.cols / 4) * 5 * 1;
    resized_img.convertTo(resized_img, CV_32FC3);

    m_results[0] = new float[score_output_size];
    m_results[1] = new float[geometry_output_size];
    float* local_input = new float[input_size];

    CHECK(cudaMalloc(&m_buffers[0], input_size * sizeof(float)));  // data
    CHECK(cudaMalloc(&m_buffers[1], score_output_size * sizeof(float)));
    CHECK(cudaMalloc(&m_buffers[2], geometry_output_size * sizeof(float)));

    float* input_ptr_pos = local_input;
    std::vector<cv::Mat> channel_splits;

    channel_splits.clear();
    for (size_t j = 0; j < 3; ++j) {
        channel_splits.emplace_back(resized_img.rows, resized_img.cols,
                                    CV_32FC1, (void*)input_ptr_pos);
        input_ptr_pos += resized_img.rows * resized_img.cols;
    }
    cv::split(resized_img, channel_splits);

    // cpu to gpu 异步
    /*CHECK(cudaMemcpyAsync(m_buffers[0], local_input,
                    input_size * sizeof(float),
    cudaMemcpyHostToDevice,m_stream)); std::cout << "start inference" <<
    std::endl;
    // do inference
    //m_context->enqueue(1, m_buffers.data(), m_stream, nullptr);
    m_context->enqueueV2(m_buffers.data(),m_stream,nullptr);
    //m_context->executeV2(m_buffers.data());
    // gpu to cpu
    CHECK(cudaMemcpyAsync(m_results[0], m_buffers[1], score_output_size *
    sizeof(float), cudaMemcpyDeviceToHost,m_stream));
    CHECK(cudaMemcpyAsync(m_results[1], m_buffers[2], geometry_output_size *
    sizeof(float), cudaMemcpyDeviceToHost,m_stream));
    // host和device同步
    CHECK(cudaStreamSynchronize(m_stream));*/

    // cpu to gpu 同步
    CHECK(cudaMemcpy(m_buffers[0], local_input, input_size * sizeof(float),
                     cudaMemcpyHostToDevice));
    // std::cout << "start inference" << std::endl;
    // do inference
    m_context->executeV2(m_buffers.data());
    // gpu to cpu
    CHECK(cudaMemcpy(m_results[0], m_buffers[1],
                     score_output_size * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_results[1], m_buffers[2],
                     geometry_output_size * sizeof(float),
                     cudaMemcpyDeviceToHost));

    delete[] local_input;
    //std::cout << "complete infer" << std::endl;
}

//后处理
std::vector<std::vector<float>> TRTEast::process(
    double f_score_threshold, std::vector<float>& final_scores,
    std::vector<float>& avg_scores)
{
    std::vector<std::pair<int, int>> yx_indexs;
    std::vector<std::vector<float>> coords;
    std::vector<float> angles;
    std::vector<float> scores;

    for (int i = 0; i < m_feature_height * m_feature_width; i++) {
        if (m_results[0][i] > f_score_threshold) {
            yx_indexs.emplace_back((i + 1) / m_feature_width,
                                   i % m_feature_width);
            scores.push_back(m_results[0][i]);
            angles.push_back(
                m_results[1][i + m_feature_height * m_feature_width * 4]);
            coords.push_back(
                {m_results[1][i],
                 m_results[1][i + m_feature_height * m_feature_width],
                 m_results[1][i + m_feature_height * m_feature_width * 2],
                 m_results[1][i + m_feature_height * m_feature_width * 3]});
        }
    }

    std::vector<std::vector<float>> temp_boxes =
        trt_east::RestoreRectangle(yx_indexs, coords, angles, scores);
    /*std::cout<<temp_boxes.size()<<std::endl;

    for(int i=0;i<temp_boxes.size();i++){
        for(int j =0;j<8;j++){
            std::cout<<temp_boxes[i][j]<<" ";
        }
        std::cout<<""<<std::endl;
    }*/

    std::vector<std::vector<float>> boxes =
        lanms::merge_quadrangle_n9(temp_boxes, scores, m_nms_threshold);
    //获取每个box的分数
    trt_east::GetScore(boxes, m_results[0], final_scores, avg_scores,
                       m_feature_width);
    for (int i = 0; i < boxes.size(); i++) {
        for (int j = 0; j < 8; j++) {
            if (j % 2 == 0) {
                boxes[i][j] /= m_ratios[1];
            } else {
                boxes[i][j] /= m_ratios[0];
            }
        }
    }
    return boxes;
}

bool TRTEast::CheckBox(std::vector<float>& box, float final_score,
                       float avg_score, double score_threshold)
{
    if (trt_east::GetDist(int(box[0]), int(box[1]), int(box[2]), int(box[3])) <
            5 ||
        trt_east::GetDist(int(box[0]), int(box[1]), int(box[6]), int(box[7])) <
            5) {
        return false;
    }
    // std::cout<<final_scores[index]<<" "<<avg_scores[index]<<std::endl;
    if (final_score < score_threshold || avg_score < m_box_threshold) {
        return false;
    }

    for (int i = 0; i < box.size(); i++) {
        if (box[i] < 0) {
            return false;
        }
    }
    return true;
}

std::vector<TRTEastResult> TRTEast::detect(cv::Mat& image,
                                           double score_threshold)
{
    this->detect_initialise();
    doInference(image);
    /*
    std::cout<<1<<std::endl;
    for(int i =0;i<m_feature_height*m_feature_width;i++){
        if(m_results[0][i]>0.8){
            std::cout<<(i+1)/m_feature_width<<" "<<i%m_feature_width<<std::endl;
        }
    }*/
    std::vector<float> final_scores;
    std::vector<float> avg_scores;
    // f_score阈值默认为0.8，和论文中一样
    std::vector<std::vector<float>> results =
        this->process(0.8, final_scores, avg_scores);
    //std::cout << m_ratios[0] << " " << m_ratios[1] << std::endl;
    int index = 0;
    for (auto& result : results) {

        if (!CheckBox(result, final_scores[index], avg_scores[index],
                      score_threshold)) {
            index++;
            continue;
        }
        TRTEastResult temp;
        // 0代表了检测的文本
        temp.label = 0;
        temp.score = final_scores[index];
        temp.box_coordinates[0] = int(result[0]);
        temp.box_coordinates[1] = int(result[1]);
        temp.box_coordinates[2] = int(result[2]);
        temp.box_coordinates[3] = int(result[3]);
        temp.box_coordinates[4] = int(result[4]);
        temp.box_coordinates[5] = int(result[5]);
        temp.box_coordinates[6] = int(result[6]);
        temp.box_coordinates[7] = int(result[7]);
        m_detection_results.push_back(temp);
        index++;
    }
    return m_detection_results;
}

std::shared_ptr<TRTEastBase> CreateEast()
{
    return std::shared_ptr<TRTEastBase>(new TRTEast());
}
