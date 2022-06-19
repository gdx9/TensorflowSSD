#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <deque>
#include <utility>
#include <optional>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include "DetectionData.h"

namespace ssd_detector{
    constexpr char kCentersPath[] = "../centers.bin";
    constexpr char kWhPath[] = "../wh.bin";
    constexpr char kNetPbPath[] = "../model_ssd_pb.pb";

    constexpr float kNormalizationRatio = 1.f / 255.f;
    constexpr size_t kElNumber = 115800;
    constexpr size_t kCentersWhLen = 123520;
    constexpr float kThreshold = 0.8;
    constexpr float kIouThreshold = 0.1;

    class DetectorTf{
    private:
        float const* const centers;
        float const* const wh;
        cv::dnn::Net model;

        float* readFile(const std::string kFilePath, const size_t kFileLen);

        float* getPredictedArray(cv::Mat& image);
        std::deque<BestBoxInfo> parsePrediction(float* input) const;

        std::deque<BoxInfo> getBestBoxes(float const * input) const;
        void calcMaxProb(std::deque<BoxInfo>& bestBoxes,
                float const * input, const size_t loc) const noexcept;

        void xywhToPoints(std::deque<BestBoxInfo>& bbis, const BoxInfo& bi) const;
        std::deque<BestBoxInfo> nonMaxSuppression(std::deque<BestBoxInfo>& bbis) const;
        [[nodiscard]]
        float IoU(const BestBoxInfo& box1, const BestBoxInfo& box2) const;

    public:
        DetectorTf();
        ~DetectorTf();

        std::optional<std::deque<BestBoxInfo>> predictImage(cv::Mat& image);

    };
}
