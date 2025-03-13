#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <stdexcept>
#include <memory>
#include "PoseTracker.h"
#include "SuitOverlay.h"

using namespace cv;
using namespace std;

class PoseEstimator {
public:
    PoseEstimator(const string &modelPath, bool multiPose = true, int inpWidth = 192, int inpHeight = 192);
    float* runInference(const Mat &inferenceImage);
    // Draws keypoints, connections, and overlays suit parts.
    void drawPosesGL(const Mat &displayImage, float* output, const SuitImages &suit,
                     const SuitScaleFactors &scales, bool drawLines);
private:
    unique_ptr<tflite::FlatBufferModel> model;
    unique_ptr<tflite::Interpreter> interpreter;
    bool multiPose;
    int inputWidth;
    int inputHeight;
    const float poseThreshold;
    const float keypointThreshold;
    PoseTracker tracker;
};

#endif // POSE_ESTIMATOR_H
