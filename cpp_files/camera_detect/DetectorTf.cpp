#include "DetectorTf.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

namespace ssd_detector{
    DetectorTf::DetectorTf()
        :centers(readFile(kCentersPath, kCentersWhLen)),
         wh(readFile(kWhPath, kCentersWhLen))
    {
        clog << "DetectorTf constructor" << endl;

        model = readNetFromTensorflow(kNetPbPath);

        // model parameters
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);// DNN_BACKEND_OPENCV
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    }

    DetectorTf::~DetectorTf(){
        clog << "DetectorTf destructor" << endl;
        delete [] centers;
        delete [] wh;
    }

    optional<deque<BestBoxInfo>> DetectorTf::predictImage(Mat& image){
        float* predArray = getPredictedArray(image);
        deque<BestBoxInfo> predBoxes = parsePrediction(predArray);

        // adjust sizes from 300x300 to 640x480 rectangles
        for(BestBoxInfo& bbi : predBoxes){
            bbi.adjustBoxesForNewSize({image.cols, image.rows});
        }

        // remove small boxes
        predBoxes.erase(remove_if(predBoxes.begin(), predBoxes.end(),
                [this](BestBoxInfo& b){
                    if(b.endPt.x - b.startPt.x < kFoundBoxMinWidthHeight//width
                    || b.endPt.y - b.startPt.y < kFoundBoxMinWidthHeight//height
                    ){
                        return true;
                    }
                    return false;
                }), predBoxes.end());

        if(predBoxes.size() > 0) return predBoxes;
        else return nullopt;
    }

    float* DetectorTf::readFile(const string kFilePath, const size_t kFileLen){
        std::ifstream fin(kFilePath, std::ios::in | std::ios::binary);

        if(!fin){
            cerr << "error in opening file" << endl;
        }else{
            // read bytes
            float* fileBytes = new float[kFileLen];
            fin.read(reinterpret_cast<char*>(fileBytes), sizeof(float)*kFileLen);
            fin.close();

            return fileBytes;

        }

        return nullptr;

    }

    float* DetectorTf::getPredictedArray(Mat& image){
        //create blob from image
        Mat blob = blobFromImage(image,
            kNormalizationRatio,
            Size(kImageSize, kImageSize));

        model.setInput(blob);

        Mat output = model.forward();

        float* ptr = output.ptr<float>();

        return ptr;

    }

    deque<BestBoxInfo> DetectorTf::parsePrediction(float* input) const{

        // get best boxes
        deque<BoxInfo> bestBoxes = getBestBoxes(input);

        deque<BestBoxInfo> bbis;

        for(auto& pr : bestBoxes){
            // calculate xywh points for every box
            xywhToPoints(bbis, pr);
        }

        deque<BestBoxInfo> predictedData = nonMaxSuppression(bbis);

        return predictedData;
    }

    deque<BoxInfo> DetectorTf::getBestBoxes(float const * input) const{

        deque<BoxInfo> bestBoxes;

        for(size_t p = 0; p < kElNumber; p += kStep){
            calcMaxProb(bestBoxes, input + p, p / kStep);
        }

        return bestBoxes;

    }

    void DetectorTf::calcMaxProb(deque<BoxInfo>& bestBoxes,
            float const * input, const size_t loc) const noexcept{
        // largest value after softmax is at the same position as before softmax
        // so we skip all the redundant calculations
        size_t maxPos = 0;
        float maxVal = 0.;

        for(size_t i = 0; i < kClassesNum; ++i){
            if(maxVal < input[i]){
                maxVal = input[i];
                maxPos = i;
            }
        }

        if(maxPos != static_cast<int>(DetectionClass::BACKGROUND)){
            float expSum = 0.;
            for(size_t i = 0; i < kClassesNum; ++i){
                expSum += exp(input[i]);
            }

            maxVal = exp(maxVal);
            float probability = maxVal / expSum;

            if(probability > kThreshold){
                bestBoxes.push_back({static_cast<DetectionClass>(maxPos), loc, probability,
                    input[kPosX],input[kPosY],
                    input[kPosW],input[kPosH]});
            }
        }
    }

    void DetectorTf::xywhToPoints(deque<BestBoxInfo>& bbis, const BoxInfo& bi) const {
        size_t xyLoc = bi.loc * 2;
        float centX = centers[xyLoc] + bi.x0;
        float centY = centers[xyLoc + 1] + bi.y0;

        float w = wh[xyLoc] + bi.w;
        float h = wh[xyLoc + 1] + bi.h;

        float x = centX - w / 2;
        float y = centY - h / 2;

        bbis.push_back({bi.cls, bi.prob, Point2i(x, y), Point2i(x + w, y + h)});

    }

    deque<BestBoxInfo> DetectorTf::nonMaxSuppression(deque<BestBoxInfo>& bbis) const{
        deque<BestBoxInfo> predictedData;
        // sort - no need, we use swap

        while(!bbis.empty()){
            // take box and check if there're similar boxes
            BestBoxInfo bl = bbis.front();
            bbis.pop_front();

            bbis.erase(remove_if(bbis.begin(),bbis.end(),
                    [&bl, this](BestBoxInfo& b) {

                if(bl.cls != b.cls) return false;

                float iou = this->IoU(bl, b);
                if(iou > kIouThreshold){
                    // receive the one with highest score
                    if(bl.prob < b.prob) swap(bl, b);

                    return true;
                }

                return false;
            }),bbis.end());

            predictedData.push_back(bl);

        }

        return predictedData;

    }

    float DetectorTf::IoU(const BestBoxInfo& box1, const BestBoxInfo& box2) const{
        int xmin = (box1.startPt.x > box2.startPt.x) ? box1.startPt.x : box2.startPt.x;
        int ymin = (box1.startPt.y > box2.startPt.y) ? box1.startPt.y : box2.startPt.y;
        int xmax = (box1.endPt.x > box2.endPt.x) ? box2.endPt.x : box1.endPt.x;
        int ymax = (box1.endPt.y > box2.endPt.y) ? box2.endPt.y : box1.endPt.y;

        int intersection = std::abs(std::max(xmax-xmin, 0) * std::max(ymax-ymin, 0));
        int boxArea1 = std::abs( (box1.endPt.x - box1.startPt.x) *
                (box1.endPt.y - box1.startPt.y));
        int boxArea2 = std::abs( (box2.endPt.x - box2.startPt.x) *
                    (box2.endPt.y - box2.startPt.y));

        int unionArea = boxArea1 + boxArea2 - intersection;
        float iou = static_cast<float>(intersection) / static_cast<float>(unionArea);

        return iou;

    }

}
