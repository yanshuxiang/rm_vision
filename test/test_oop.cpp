#include "../include/finder.h"
#include "../include/visualizer.h"
#include<opencv2/opencv.hpp>
#include"../include/camera.h"

int main() {
    
    // 实例化对象
    ArmorDetector detector;
    Visualizer visualizer;
    Camera cam;
    
    cam.init();
    cv::Mat frame;
    std::vector<std::pair<LightBar, LightBar>> history;

    while (true) {
        // 1. 让检测器去工作
        cam.getImage(frame);
        auto current_armors = detector.detect(frame);
        
        // 2. 更新记忆状态
        bool is_lost = current_armors.empty();
        if (!is_lost) history = current_armors;

        // 3. 让渲染器去绘图
        visualizer.render(frame, current_armors, history, is_lost);

        cv::imshow("OOP RoboMaster", frame);
        if (cv::waitKey(16) == 'q') break;
    }
    return 0;
}