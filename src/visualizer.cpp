#include "../include/visualizer.h"

Visualizer::Visualizer() {}

void Visualizer::render(cv::Mat& output, 
                        const std::vector<std::pair<LightBar, LightBar>>& current,
                        const std::vector<std::pair<LightBar, LightBar>>& history,
                        bool is_lost) {
    
    // 1. 绘制左上角信息面板背景 (HUD Background)
    const auto& display_data = is_lost ? history : current;
    drawHUD(output, display_data, is_lost);

    // 2. 绘制实时的装甲板几何框
    for (const auto& armor : current) {
        drawArmorBox(output, armor);
    }
}

void Visualizer::drawArmorBox(cv::Mat& img, const std::pair<LightBar, LightBar>& armor) {
    // 计算装甲板中心
    cv::Point center((armor.first.center.x + armor.second.center.x) / 2,
                     (armor.first.center.y + armor.second.center.y) / 2);
    
    // 确定颜色
    cv::Scalar color = (armor.first.color == ArmorColor::RED) ? RED_COLOR : BLUE_COLOR;

    // 提取两个灯条的所有顶点用于计算凸包 (Convex Hull)
    cv::Point2f pts1[4], pts2[4];
    armor.first.rect.points(pts1);
    armor.second.rect.points(pts2);
    
    std::vector<cv::Point> all_pts;
    for(int i = 0; i < 4; ++i) {
        all_pts.push_back(cv::Point(pts1[i].x, pts1[i].y));
        all_pts.push_back(cv::Point(pts2[i].x, pts2[i].y));
    }

    // 绘制装甲板外框
    std::vector<cv::Point> hull;
    cv::convexHull(all_pts, hull);
    cv::polylines(img, std::vector<std::vector<cv::Point>>{hull}, true, color, 2);

    // 绘制中心点和标签
    cv::circle(img, center, 5, color, -1);
    std::string label = (armor.first.color == ArmorColor::RED) ? "RED" : "BLUE";
    cv::putText(img, label, cv::Point(center.x - 50, center.y - 40), 
                cv::FONT_HERSHEY_DUPLEX, 1.0, color, 2);
}

void Visualizer::drawHUD(cv::Mat& img, const std::vector<std::pair<LightBar, LightBar>>& armors, bool is_lost) {
    if (armors.empty()) return;

    // 动态计算背景框高度
    int base_y = 70;
    int line_height = 60;
    int bg_h = 20 + armors.size() * line_height;
    
    // 绘制半透明黑色背景
    cv::Rect bg_rect(10, 10, 600, bg_h);
    bg_rect &= cv::Rect(0, 0, img.cols, img.rows); // 防止越界
    cv::Mat roi = img(bg_rect);
    cv::Mat black_mask(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.5, black_mask, 0.5, 0, roi);

    // 绘制列表信息
    for (size_t i = 0; i < armors.size(); ++i) {
        const auto& a = armors[i];
        cv::Scalar color = (a.first.color == ArmorColor::RED) ? RED_COLOR : BLUE_COLOR;
        std::string color_name = (a.first.color == ArmorColor::RED) ? "RED" : "BLUE";
        
        cv::Point center((a.first.center.x + a.second.center.x) / 2,
                         (a.first.center.y + a.second.center.y) / 2);

        std::string info = cv::format("[%zu] %s: (%d, %d)", i + 1, color_name.c_str(), center.x, center.y);
        if (is_lost) info += " [LOST]";

        cv::putText(img, info, cv::Point(30, base_y + i * line_height), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 2);
    }
}