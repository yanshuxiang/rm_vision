cmake_minimum_required(VERSION 3.10)
project(rm_vision)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 1. 查找 OpenCV
find_package(OpenCV REQUIRED)

# 2. 设置 SDK 路径
set(SDK_PATH "/opt/HuarayTech/MVviewer")

# 3. 包含头文件目录
include_directories(
    "${SDK_PATH}/include"
    ${OpenCV_INCLUDE_DIRS}
)


# 5. 生成可执行程序
add_executable(app 
    test/test_camera.cpp
    src/camera.cpp
    )

# 6. 链接库文件 (根据你的 lib 目录截图，库名为 MVSDK)
# 同时链接图像转换库 libImageConvert.so，这是显示画面所必须的
target_link_libraries(app
    "${SDK_PATH}/lib/libMVSDK.so"
    "${SDK_PATH}/lib/libImageConvert.so"
    ${OpenCV_LIBS}
)

# 7. 设置运行时搜索路径 (RPATH)
# 这样运行 app 时会自动去 SDK 目录找库，不需要 export LD_LIBRARY_PATH
set_target_properties(app PROPERTIES 
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${SDK_PATH}/lib"
)