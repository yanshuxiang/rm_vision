#ifndef FINDER_H
#define FINDER_H

#include <opencv2/opencv.hpp>
#include <vector>

enum class ArmorColor { RED, BLUE, UNKNOWN };

struct LightBar {
    cv::Point2f center;
    cv::Size2f size;
    float angle;
    cv::RotatedRect rect;
    ArmorColor color; 
};

class ArmorDetector {
public:
    ArmorDetector();
    ~ArmorDetector() = default;

    // 核心外部接口：传入图像，返回检测到的装甲板对
    std::vector<std::pair<LightBar, LightBar>> detect(const cv::Mat& frame);

private:
    // 内部处理步骤（私有辅助函数）
    cv::Mat preprocess(const cv::Mat& frame);
    ArmorColor determineColor(const cv::Mat& frame, const std::vector<cv::Point>& contour);
    float calculateSideAngle(const cv::RotatedRect& rect, float& h, float& w);
    bool isMatchingPair(const LightBar& b1, const LightBar& b2);

    // 算法参数（私有常量）
    static constexpr int BRIGHTNESS_THRESHOLD = 180;
    static constexpr float MIN_ASPECT_RATIO = 2.5f;
    static constexpr float MAX_ANGLE_DIFF = 5.0f;
};

#endif