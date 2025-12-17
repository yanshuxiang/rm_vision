#include<iostream>
#include<opencv2/opencv.hpp>

int main(){
    cv::VideoCapture cap("../video/1.mp4");
    if(!cap.isOpened()){
        return -1;
    }
    double fps=cap.get(cv::CAP_PROP_FPS);
    std::cout<<"fps="<<fps<<std::endl;
    cv::Mat frame;
    while (1){
        bool isSuccess=cap.read(frame);
        if (!isSuccess || frame.empty()) {
                std::cout << "视频播放结束" << std::endl;
                break;
            }
            cv::imshow("Video Player", frame);
        if (cv::waitKey(1000/fps) == 'q') {
            break;
        }
    }

    // 6. 释放并关闭窗口
    cap.release();
    cv::destroyAllWindows();
    return 0;
}