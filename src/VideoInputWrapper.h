#ifndef VIDEO_INPUT_WRAPPER_H
#define VIDEO_INPUT_WRAPPER_H

#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

class VideoInputWrapper {
public:
    VideoInputWrapper(const string &source);
    bool getFrame(Mat &frame);
    double getFrameWidth() const;
    double getFrameHeight() const;
private:
    VideoCapture cap;
    bool isCamera;
};

#endif // VIDEO_INPUT_WRAPPER_H
