#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// --- 常量定义 ---
const int MIN_CONTOUR_AREA = 50;
const float MIN_ASPECT_RATIO = 2.5f;
const float MAX_ANGLE_DIFF = 5.0f;
const float MAX_Y_DIFF_RATIO = 0.3f;
const float MIN_LENGTH_SIMILARITY = 0.75f;
const float MIN_DISTANCE_RATIO = 0.8f;
const float MAX_DISTANCE_RATIO = 5.0f;
const float MAX_ALIGNMENT_ERROR = 10.0f;
const float MAX_LONGSIDE_DEGREE = 95.0f;
const float MIN_LONGSIDE_DEGREE = 85.0f;
const int BRIGHTNESS_THRESHOLD = 180;
const int FPS = 30;

enum ArmorColor { RED, BLUE, UNKNOWN };

struct LightBar {
    Point2f center;
    Size2f size;
    float angle;
    RotatedRect rect;
    ArmorColor color; 
};

/**
 * 获取灯条颜色
 */
ArmorColor getColor(const Mat& bgr_img, const vector<Point>& contour) {
    long long red_sum = 0;
    long long blue_sum = 0;
    for (const auto& pt : contour) {
        Vec3b color = bgr_img.at<Vec3b>(pt);
        blue_sum += color[0];
        red_sum += color[2];
    }
    return (blue_sum > red_sum) ? BLUE : RED;
}

/**
 * 计算旋转矩形长边的角度
 */
float calculateSideAngle(const RotatedRect& rect, float& height, float& width) {
    Point2f pts[4];
    rect.points(pts);
    float d1 = norm(pts[0] - pts[1]);
    float d2 = norm(pts[1] - pts[2]);
    Point2f p_a, p_b;
    if (d1 >= d2) {
        height = d1; width = d2;
        p_a = pts[0]; p_b = pts[1];
    } else {
        height = d2; width = d1;
        p_a = pts[1]; p_b = pts[2];
    }
    float delta_x = p_b.x - p_a.x;
    float delta_y = -(p_b.y - p_a.y); 
    float angle_degrees = atan2(delta_y, delta_x) * 180.0 / CV_PI;
    while (angle_degrees < 0) angle_degrees += 360.0;
    return fmod(angle_degrees, 180.0f);
}

/**
 * 图像预处理
 */
Mat initImg(const Mat& img) {
    Mat gray, bright_mask, processed_mask;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, bright_mask, BRIGHTNESS_THRESHOLD, 255, THRESH_BINARY);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(bright_mask, processed_mask, kernel);
    return processed_mask;
}

/**
 * 检测装甲板
 */
vector<pair<LightBar, LightBar>> detect(const Mat& bgr_frame, const Mat& processed_mask) {
    vector<vector<Point>> contours;
    findContours(processed_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<LightBar> candidates;
    for (const auto& contour : contours) {
        if (contourArea(contour) < MIN_CONTOUR_AREA) continue;
        RotatedRect raw_rect = minAreaRect(contour);
        float h, w;
        float angle = calculateSideAngle(raw_rect, h, w);
        float ratio = (w > 0) ? (h / w) : 0;
        if (ratio < MIN_ASPECT_RATIO) continue;
        if (angle < MIN_LONGSIDE_DEGREE || angle > MAX_LONGSIDE_DEGREE) continue;
        ArmorColor col = getColor(bgr_frame, contour);
        candidates.push_back({raw_rect.center, Size2f(w, h), angle, raw_rect, col});
    }
    sort(candidates.begin(), candidates.end(), [](const LightBar& a, const LightBar& b) {
        return a.center.x < b.center.x;
    });
    vector<pair<LightBar, LightBar>> final_armors;
    if (candidates.size() < 2) return final_armors;
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            const auto& b1 = candidates[i];
            const auto& b2 = candidates[j];
            if (b1.color != b2.color) continue;
            float avg_h = (b1.size.height + b2.size.height) / 2.0f;
            if (abs(b1.angle - b2.angle) > MAX_ANGLE_DIFF) continue;
            float h_min = min(b1.size.height, b2.size.height);
            float h_max = max(b1.size.height, b2.size.height);
            if (h_min / h_max < MIN_LENGTH_SIMILARITY) continue;
            if (abs(b1.center.y - b2.center.y) / avg_h > MAX_Y_DIFF_RATIO) continue;
            float dist = norm(b1.center - b2.center);
            float dist_ratio = dist / avg_h;
            if (dist_ratio < MIN_DISTANCE_RATIO || dist_ratio > MAX_DISTANCE_RATIO) continue;
            float dy = b2.center.y - b1.center.y;
            float dx = b2.center.x - b1.center.x;
            float angle_conn = abs(atan2(dy, dx) * 180.0 / CV_PI);
            if (angle_conn > 90) angle_conn = 180.0f - angle_conn;
            if (angle_conn > MAX_ALIGNMENT_ERROR) continue;
            final_armors.push_back({b1, b2});
        }
    }
    return final_armors;
}

/**
 * 绘图函数
 */
void drawResults(Mat& img, const vector<pair<LightBar, LightBar>>& current_armors, 
                 const vector<pair<LightBar, LightBar>>& display_armors, bool is_history) {
    
    // 1. 绘制左上角 HUD 背景 (根据显示行数动态拉长)
    if (!display_armors.empty()) {
        int bg_h = 40 + display_armors.size() * 60;
        Rect bg_rect(10, 10, 650, bg_h);
        bg_rect &= Rect(0, 0, img.cols, img.rows);
        
        Mat roi = img(bg_rect);
        Mat black_bg(roi.size(), CV_8UC3, Scalar(0, 0, 0));
        addWeighted(roi, 0.4, black_bg, 0.6, 0, roi); 
    }

    // 2. 绘制当前帧的装甲板实线框
    for (const auto& armor : current_armors) {
        Point center((armor.first.center.x + armor.second.center.x) / 2,
                     (armor.first.center.y + armor.second.center.y) / 2);
        Scalar color_bgr = (armor.first.color == RED) ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
        
        Point2f pts1[4], pts2[4];
        armor.first.rect.points(pts1);
        armor.second.rect.points(pts2);
        vector<Point> all_pts;
        for(int j=0; j<4; ++j) {
            all_pts.push_back(Point(pts1[j].x, pts1[j].y));
            all_pts.push_back(Point(pts2[j].x, pts2[j].y)); // 这里已修正为 .x
        }
        vector<Point> hull;
        convexHull(all_pts, hull);
        polylines(img, vector<vector<Point>>{hull}, true, color_bgr, 3);
        circle(img, center, 6, color_bgr, -1);
        
        string t = (armor.first.color == RED) ? "RED" : "BLUE";
        putText(img, t, Point(center.x - 55, center.y - 50), FONT_HERSHEY_DUPLEX, 1.4, color_bgr, 3);
    }

    // 3. 绘制左上角记忆信息
    int y_offset = 70;
    for (size_t i = 0; i < display_armors.size(); ++i) {
        const auto& armor = display_armors[i];
        Point center((armor.first.center.x + armor.second.center.x) / 2,
                     (armor.first.center.y + armor.second.center.y) / 2);
        
        Scalar color_bgr = (armor.first.color == RED) ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
        string color_str = (armor.first.color == RED) ? "RED" : "BLUE";
        
        // 拼接字符串：如果是历史数据则加上 [LOST]
        string info = format("[%zu] %s: (%d, %d)", i + 1, color_str.c_str(), center.x, center.y);
        if (is_history) info += " [LOST]";

        // 左上角文字：缩放比例 1.5，粗细 3
        putText(img, info, Point(30, y_offset), FONT_HERSHEY_SIMPLEX, 1.5, color_bgr, 3);
        y_offset += 60;
    }
}

int main() {
    string video_path = "../video/2.mp4"; // 请确保路径正确
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "无法打开视频文件" << endl;
        return -1;
    }

    Mat frame, processed_mask;
    vector<pair<LightBar, LightBar>> last_detected_armors;

    while (cap.read(frame)) {
        processed_mask = initImg(frame);
        auto current_armors = detect(frame, processed_mask);
        
        bool is_history = false;
        if (!current_armors.empty()) {
            last_detected_armors = current_armors;
            is_history = false;
        } else {
            is_history = true;
        }

        drawResults(frame, current_armors, last_detected_armors, is_history);

        Mat resized_img;
        resize(frame, resized_img, Size(frame.cols / 2, frame.rows / 2), 0, 0, INTER_CUBIC);
        imshow("RoboMaster", resized_img);
        
        if (waitKey(1000 / FPS) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}