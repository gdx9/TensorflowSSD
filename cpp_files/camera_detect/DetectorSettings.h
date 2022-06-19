#pragma once

constexpr int kCameraImageResizedWidth  = 640;
constexpr int kCameraImageResizedHeight = 480;
constexpr int kFrameDurationMs = 10;
constexpr char kExitKey = 'q';

namespace ssd_detector{
    constexpr size_t kImageSize = 300;
    constexpr int kFoundBoxMinWidthHeight = 25;

    enum class DetectionClass{
        CLOCK      = 0,
        APPLE      = 1,
        BOTTLE     = 2,
        CAT        = 3,
        CUP        = 4,
        KEY        = 5,
        SPIDER     = 6,
        BACKGROUND = 7
    };

    constexpr const char* DetectionClassToString(const DetectionClass dc) noexcept {
        switch(dc){
            case DetectionClass::CLOCK:  return "CLOCK";
            case DetectionClass::APPLE:  return "APPLE";
            case DetectionClass::BOTTLE: return "BOTTLE";
            case DetectionClass::CAT:    return "CAT";
            case DetectionClass::CUP:    return "CUP";
            case DetectionClass::KEY:    return "KEY";
            case DetectionClass::SPIDER: return "SPIDER";
            default:                     return "BACKGROUND";
        }
    }

    constexpr size_t kClassesNum = static_cast<size_t>(DetectionClass::BACKGROUND) + 1;// 8
    constexpr size_t kStep = kClassesNum + 4;// 4 -> x,y,w,h
    // positions inside box
    constexpr size_t kPosX = kClassesNum;// 8
    constexpr size_t kPosY = kPosX + 1;// 9
    constexpr size_t kPosW = kPosY + 1;// 10
    constexpr size_t kPosH = kPosW + 1;// 11

}
