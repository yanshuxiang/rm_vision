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
const int FPS = 60;

// 灯条结构体
struct LightBar {
    Point2f center;
    Size2f size;
    float angle;
    RotatedRect rect;
};

/**
 * 计算旋转矩形长边的角度
 * 返回值范围 [0, 180)
 */
float calculateSideAngle(const RotatedRect& rect, float& height, float& width) {
    Point2f pts[4];
    rect.points(pts);

    // 确定哪边是长边
    float d1 = norm(pts[0] - pts[1]);
    float d2 = norm(pts[1] - pts[2]);

    Point2f p_a, p_b;
    if (d1 >= d2) {
        height = d1;
        width = d2;
        p_a = pts[0];
        p_b = pts[1];
    } else {
        height = d2;
        width = d1;
        p_a = pts[1];
        p_b = pts[2];
    }

    // 计算角度 (使用与 Python 相同的逻辑)
    float delta_x = p_b.x - p_a.x;
    float delta_y = -(p_b.y - p_a.y); // 图像坐标系 y 向下，需取反

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
 * 灯条检测与装甲板匹配
 */
vector<pair<LightBar, LightBar>> detect(const Mat& processed_mask) {
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

        candidates.push_back({raw_rect.center, Size2f(w, h), angle, raw_rect});
    }

    // 按 X 坐标排序
    sort(candidates.begin(), candidates.end(), [](const LightBar& a, const LightBar& b) {
        return a.center.x < b.center.x;
    });

    vector<pair<LightBar, LightBar>> final_armors;
    if (candidates.size() < 2) return final_armors;

    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            const auto& b1 = candidates[i];
            const auto& b2 = candidates[j];

            float avg_h = (b1.size.height + b2.size.height) / 2.0f;

            // 1. 平行度
            if (abs(b1.angle - b2.angle) > MAX_ANGLE_DIFF) continue;

            // 2. 长度相似度
            float h_min = min(b1.size.height, b2.size.height);
            float h_max = max(b1.size.height, b2.size.height);
            if (h_min / h_max < MIN_LENGTH_SIMILARITY) continue;

            // 3. 中心高度差
            if (abs(b1.center.y - b2.center.y) / avg_h > MAX_Y_DIFF_RATIO) continue;

            // 4. 距离比例
            float dist = norm(b1.center - b2.center);
            float dist_ratio = dist / avg_h;
            if (dist_ratio < MIN_DISTANCE_RATIO || dist_ratio > MAX_DISTANCE_RATIO) continue;

            // 5. 连线水平夹角
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
 * 绘制结果
 */
void drawRec(Mat& img, const vector<pair<LightBar, LightBar>>& armors) {
    for (const auto& armor : armors) {
        Point2f pts1[4], pts2[4];
        armor.first.rect.points(pts1);
        armor.second.rect.points(pts2);

        vector<Point> all_pts;
        for(int i=0; i<4; ++i) {
            all_pts.push_back(Point(pts1[i].x, pts1[i].y));
            all_pts.push_back(Point(pts2[i].x, pts2[i].y));
        }

        vector<Point> hull;
        convexHull(all_pts, hull);
        
        vector<vector<Point>> hulls = {hull};
        polylines(img, hulls, true, Scalar(0, 255, 0), 2);

        Point center((armor.first.center.x + armor.second.center.x) / 2,
                     (armor.first.center.y + armor.second.center.y) / 2);
        circle(img, center, 4, Scalar(0, 0, 255), -1);
    }
}

int main() {
    string video_path = "../video/1.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "无法打开视频文件" << endl;
        return -1;
    }

    Mat frame, processed;
    while (cap.read(frame)) {
        processed = initImg(frame);
        auto armors = detect(processed);
        drawRec(frame, armors);

        Mat resized_img;
        resize(frame, resized_img, Size(frame.cols / 2, frame.rows / 2), 0, 0, INTER_CUBIC);
        
        imshow("img", resized_img);
        if (waitKey(1000 / FPS) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}