cmake_minimum_required(VERSION 3.26)
project(webcam_multipose)

find_package(box2d)
find_package(tensorflowlite)
find_package(OpenCV)
find_package(SDL2 REQUIRED CONFIG)
find_package(opengl_system REQUIRED CONFIG)

add_executable(webcam_multipose main.cpp
        ball.cpp
        ball.h)

target_link_libraries(webcam_multipose tensorflow::tensorflowlite opencv::opencv box2d::box2d SDL2::SDL2main SDL2::SDL2-static opengl::opengl)
