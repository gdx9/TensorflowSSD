#include "DetectorTf.h"

int main(){
    cv::Mat frame;
    cv::VideoCapture cap;

    std::unique_ptr<ssd_detector::DetectorTf>
                        md = std::make_unique<ssd_detector::DetectorTf>();

    cap.open(0, cv::CAP_ANY);

    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera" << std::endl;
        return EXIT_FAILURE;
    }

    while(true){
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed" << std::endl;
            break;
        }

        cv::resize(frame, frame,
                cv::Size(::kCameraImageResizedWidth, ::kCameraImageResizedHeight));

        // predict via Net
        std::optional<std::deque<ssd_detector::BestBoxInfo>>
                                    predictedData = md->predictImage(frame);

        if(std::nullopt != predictedData){
            for(ssd_detector::BestBoxInfo& bbi : predictedData.value()){
                bbi.drawBoxDataOnMat(frame);
            }
        }

        cv::imshow("frame", frame);
        char key = cv::waitKey(::kFrameDurationMs);
        if(key == ::kExitKey || key == 27){// 27 - Esc
            std::clog << "Exit key pressed" << std::endl;
            break;
        }
    }

    cv::destroyAllWindows();

    std::clog << "program end" << std::endl;

    return EXIT_SUCCESS;

}
