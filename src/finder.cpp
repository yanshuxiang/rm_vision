#include "../include/finder.h"
#include <cmath>
#include <algorithm>

ArmorDetector::ArmorDetector() {}

// 1. 预处理
cv::Mat ArmorDetector::preprocess(const cv::Mat& frame) {
    cv::Mat gray, mask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, BRIGHTNESS_THRESHOLD, 255, cv::THRESH_BINARY);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    return mask;
}

// 2. 计算角度（必须带 ArmorDetector::）
float ArmorDetector::calculateSideAngle(const cv::RotatedRect& rect, float& h, float& w) {
    cv::Point2f pts[4];
    rect.points(pts);
    float d1 = cv::norm(pts[0] - pts[1]);
    float d2 = cv::norm(pts[1] - pts[2]);
    cv::Point2f p_a, p_b;
    if (d1 >= d2) {
        h = d1; w = d2;
        p_a = pts[0]; p_b = pts[1];
    } else {
        h = d2; w = d1;
        p_a = pts[1]; p_b = pts[2];
    }
    float delta_x = p_b.x - p_a.x;
    float delta_y = -(p_b.y - p_a.y); 
    float angle_degrees = atan2(delta_y, delta_x) * 180.0 / CV_PI;
    while (angle_degrees < 0) angle_degrees += 360.0;
    return fmod(angle_degrees, 180.0f);
}

// 3. 确定颜色（必须带 ArmorDetector::）
ArmorColor ArmorDetector::determineColor(const cv::Mat& frame, const std::vector<cv::Point>& contour) {
    long long red_sum = 0;
    long long blue_sum = 0;
    for (const auto& pt : contour) {
        cv::Vec3b color = frame.at<cv::Vec3b>(pt);
        blue_sum += color[0]; // B
        red_sum += color[2];  // R
    }
    return (blue_sum > red_sum) ? ArmorColor::BLUE : ArmorColor::RED;
}

// 4. 匹配逻辑（必须带 ArmorDetector::）
bool ArmorDetector::isMatchingPair(const LightBar& b1, const LightBar& b2) {
    if (b1.color != b2.color) return false;
    
    float avg_h = (b1.size.height + b2.size.height) / 2.0f;
    // 角度差
    if (std::abs(b1.angle - b2.angle) > MAX_ANGLE_DIFF) return false;
    // 长度相似度
    float h_min = std::min(b1.size.height, b2.size.height);
    float h_max = std::max(b1.size.height, b2.size.height);
    if (h_min / h_max < 0.75f) return false;
    // 中心 y 差
    if (std::abs(b1.center.y - b2.center.y) / avg_h > 0.3f) return false;
    // 距离比例
    float dist = cv::norm(b1.center - b2.center);
    float dist_ratio = dist / avg_h;
    if (dist_ratio < 0.8f || dist_ratio > 5.0f) return false;

    return true;
}

// 5. 主检测流程
std::vector<std::pair<LightBar, LightBar>> ArmorDetector::detect(const cv::Mat& frame) {
    cv::Mat processed = preprocess(frame);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> candidates;
    for (const auto& cnt : contours) {
        if (cv::contourArea(cnt) < 50) continue;
        cv::RotatedRect r = cv::minAreaRect(cnt);
        float h, w;
        float angle = calculateSideAngle(r, h, w); // 调用成员函数
        
        // 注意：这里的比例和角度判断使用了 static constexpr 成员
        if ((h/w) < MIN_ASPECT_RATIO) continue;
        
        candidates.push_back({r.center, cv::Size2f(w, h), angle, r, determineColor(frame, cnt)});
    }

    std::vector<std::pair<LightBar, LightBar>> results;
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (isMatchingPair(candidates[i], candidates[j])) {
                results.push_back({candidates[i], candidates[j]});
            }
        }
    }
    return results;
}