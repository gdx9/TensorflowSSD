#include "DetectionData.h"

using namespace std;
using namespace cv;

namespace ssd_detector{
    Scalar DetectionClassColor(const DetectionClass dc) {
        switch(dc){
            case DetectionClass::CLOCK:  return {0,    0,255};// 0
            case DetectionClass::APPLE:  return {255,  0,255};// 1
            case DetectionClass::BOTTLE: return {0,  255,255};// 2
            case DetectionClass::CAT:    return {0,  191,255};// 3
            case DetectionClass::CUP:    return {0,  255,  0};// 4
            case DetectionClass::KEY:    return {240,255,240};// 5
            case DetectionClass::SPIDER: return {255,255,  0};// 6
            default:                     return {128,  1,  0};// 7
        }
    }

    BestBoxInfo::BestBoxInfo(DetectionClass cls_, float prob_,
            Point2i startPt_, Point2i endPt_)
    : cls(cls_), prob(prob_), startPt(startPt_), endPt(endPt_)
    {}

    ostream& operator<<(ostream& os, const BestBoxInfo& b){
        os << DetectionClassToString(b.cls) << ", " << b.prob << ", "
                << b.startPt.x << ' ' << b.startPt.y << ", "
                << b.endPt.x << ' ' << b.endPt.y;
        return os;
    }

    void BestBoxInfo::adjustBoxesForNewSize(const Size newSize, const Size oldSize){
        this->startPt.x = this->startPt.x * newSize.width / oldSize.width;
        this->startPt.y = this->startPt.y * newSize.height / oldSize.height;

        this->endPt.x = this->endPt.x * newSize.width / oldSize.width;
        this->endPt.y = this->endPt.y * newSize.height / oldSize.height;

    }

    void BestBoxInfo::drawBoxDataOnMat(Mat& img) const{
        rectangle(img, this->startPt, this->endPt, DetectionClassColor(this->cls), 1);
        rectangle(img, this->startPt, {this->endPt.x, this->startPt.y + 16},
                DetectionClassColor(this->cls), -1);

        stringstream boxDataStream;
        boxDataStream << DetectionClassToString(this->cls) << " "
                << std::fixed << std::setprecision(2) << (this->prob * 100)
                << "%";
        putText(img, boxDataStream.str(), {this->startPt.x + 2, this->startPt.y + 12},
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0),
                1.4, LINE_4, false);

    }

    BoxInfo::BoxInfo(DetectionClass cls_, size_t loc_, float prob_, float x0_, float y0_, float w_, float h_)
            : cls(cls_), loc(loc_), prob(prob_), x0(x0_), y0(y0_), w(w_),h(h_){}

    BoxInfo::BoxInfo(BoxInfo&& b)
            :cls(std::exchange(b.cls, DetectionClass::BACKGROUND)),
            loc(std::exchange(b.loc, 0)),
            prob(std::exchange(b.prob, 0)),
            x0(std::exchange(b.x0, 0)),
            y0(std::exchange(b.y0, 0)),
            w(std::exchange(b.w, 0)),
            h(std::exchange(b.h, 0)){};

    ostream& operator<<(ostream& os, const BoxInfo& info){
        os << "class: " << DetectionClassToString(info.cls) << ", "
            << "prob: " << info.prob
            << ", coords: "
            << info.x0 << ' '
            << info.y0 << ' '
            << info.w << ' '
            << info.h
            << ", loc: " << info.loc;

        return os;
    }
}
