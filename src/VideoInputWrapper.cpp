#include "VideoInputWrapper.h"

VideoInputWrapper::VideoInputWrapper(const string &source) {
    try {
        int device = stoi(source);
        isCamera = true;
        cap.open(device);
    } catch (const exception &e) {
        isCamera = false;
        cap.open(source);
    }
    if (!cap.isOpened())
        throw runtime_error("Failed to open video source: " + source);
}

bool VideoInputWrapper::getFrame(Mat &frame) {
    cap >> frame;
    if (frame.empty() && !isCamera) {
        cap.set(CAP_PROP_POS_FRAMES, 0);
        cap >> frame;
    }
    return !frame.empty();
}

double VideoInputWrapper::getFrameWidth() const {
    return cap.get(CAP_PROP_FRAME_WIDTH);
}

double VideoInputWrapper::getFrameHeight() const {
    return cap.get(CAP_PROP_FRAME_HEIGHT);
}
