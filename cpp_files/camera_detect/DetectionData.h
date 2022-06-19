#pragma once
#include <opencv2/opencv.hpp>
#include "DetectorSettings.h"

namespace ssd_detector{
    cv::Scalar DetectionClassColor(const DetectionClass dc);

    class BestBoxInfo{
    public:
        DetectionClass cls;
        float prob;
        cv::Point2i startPt;
        cv::Point2i endPt;

        BestBoxInfo() = delete;
        BestBoxInfo(DetectionClass cls_, float prob_,
                cv::Point2i startPt_, cv::Point2i endPt_);

        void adjustBoxesForNewSize(const cv::Size newSize,
                const cv::Size oldSize={kImageSize, kImageSize});
        void drawBoxDataOnMat(cv::Mat& img) const;

        friend std::ostream& operator<<(std::ostream& os, const BestBoxInfo& b);

    };

    class BoxInfo{
    public:
        DetectionClass cls = DetectionClass::BACKGROUND;
        size_t loc = 0;
        float prob = 0.f;
        float x0 = 0.f;
        float y0 = 0.f;
        float w = 0.f;
        float h = 0.f;

        BoxInfo() = delete;
        BoxInfo(DetectionClass, size_t, float, float, float, float, float);
        BoxInfo(const BoxInfo&) = default;
        BoxInfo(BoxInfo&& b);

        friend std::ostream& operator<<(std::ostream& os, const BoxInfo& info);

    };

}
