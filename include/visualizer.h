#ifndef DRAWER_H
#define DRAWER_H

#include <opencv2/opencv.hpp>
#include "../include/finder.h" // 引用 ArmorColor 和 LightBar 定义

class Visualizer {
public:
    /**
     * @brief 构造函数，可以初始化默认颜色或字体
     */
    Visualizer();

    /**
     * @brief 渲染所有视觉效果
     * @param output 需要绘制的画布（原图）
     * @param current 当前帧检测到的装甲板
     * @param history 记忆中的装甲板
     * @param is_lost 当前帧是否丢失目标
     */
    void render(cv::Mat& output, 
                const std::vector<std::pair<LightBar, LightBar>>& current,
                const std::vector<std::pair<LightBar, LightBar>>& history,
                bool is_lost);

private:
    // 内部私有绘图辅助函数
    void drawArmorBox(cv::Mat& img, const std::pair<LightBar, LightBar>& armor);
    void drawHUD(cv::Mat& img, const std::vector<std::pair<LightBar, LightBar>>& armors, bool is_lost);
    
    // 颜色常量
    const cv::Scalar RED_COLOR = cv::Scalar(0, 0, 255);
    const cv::Scalar BLUE_COLOR = cv::Scalar(255, 0, 0);
    const cv::Scalar WHITE_COLOR = cv::Scalar(255, 255, 255);
};

#endif // DRAWER_H