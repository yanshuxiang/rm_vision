#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/camera.h"

int main() {
    Camera camera;
    camera.init();
    cv::Mat frame;
    while (true) {
        camera.getImage(frame);
        cv::imshow("window", frame);
        char key = cv::waitKey(1); // 必须有！
        if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
            break;
        }
    }

    cv::destroyAllWindows(); // 可选：显式关闭窗口
    return 0;
}