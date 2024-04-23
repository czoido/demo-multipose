cmake_minimum_required(VERSION 3.26)
project(webcam_multipose)

set(CMAKE_CXX_STANDARD 20)

find_package(box2d)
find_package(tensorflowlite)
find_package(OpenCV)

add_executable(webcam_multipose main.cpp
        ball.cpp
        ball.h)

target_link_libraries(webcam_multipose tensorflow::tensorflowlite opencv::opencv box2d::box2d)
