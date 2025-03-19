#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "PoseEstimator.h"
#include <opencv2/opencv.hpp>
#include <SDL_opengl.h>
#include <cmath>

using namespace cv;

// Global constants for drawing keypoints and connections.
const vector<Scalar> g_colors = {
        Scalar(255, 255, 0),
        Scalar(255, 0, 255),
        Scalar(0, 255, 255),
        Scalar(255, 0, 0),
        Scalar(0, 255, 0),
        Scalar(0, 0, 255)
};

const vector<pair<int, int>> g_connections = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8},
        {8, 10}, {5, 11}, {6, 12}, {11, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

PoseEstimator::PoseEstimator(const string &modelPath, bool multiPose, int inpWidth, int inpHeight)
        : multiPose(multiPose), inputWidth(inpWidth), inputHeight(inpHeight),
          poseThreshold(0.2f), keypointThreshold(0.2f)
{
    model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model)
        throw runtime_error("Failed to load model from " + modelPath);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if (interpreter->AllocateTensors() != kTfLiteOk)
        throw runtime_error("Failed to allocate tensors");
    if (multiPose) {
        interpreter->ResizeInputTensor(0, {1, inputHeight, inputWidth, 3});
        if (interpreter->AllocateTensors() != kTfLiteOk)
            throw runtime_error("Failed to reallocate tensors");
    }
}

float* PoseEstimator::runInference(const Mat &inferenceImage) {
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), inferenceImage.data,
           inferenceImage.total() * inferenceImage.elemSize());
    if (interpreter->Invoke() != kTfLiteOk)
        cerr << "Inference failed" << endl;
    return interpreter->typed_output_tensor<float>(0);
}

void PoseEstimator::drawPosesGL(const Mat &displayImage, float* output, const SuitImages &suit,
                                const SuitScaleFactors &scales, bool drawLines)
{
    int dispWidth = displayImage.cols;
    int dispHeight = displayImage.rows;
    float scaleX = static_cast<float>(dispWidth) / static_cast<float>(inputWidth);
    float scaleY = static_cast<float>(dispHeight) / static_cast<float>(inputHeight);
    int numPoses = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];

    vector<int> poseIds = tracker.trackPoses(output, numPoses, inputWidth, inputHeight, poseThreshold, keypointThreshold);

    if (drawLines) {
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        for (int p = 0; p < numPoses; p++) {
            const float* pose = output + (56 * p);
            float score = pose[55];
            if (score < poseThreshold)
                continue;
            int id = poseIds[p];
            Scalar poseColor = g_colors[(id >= 0 ? id : p) % g_colors.size()];
            glColor3f(poseColor[2] / 255.0f, poseColor[1] / 255.0f, poseColor[0] / 255.0f);
            for (int k = 0; k < 17; k++) {
                const float* keypoint = pose + 3 * k;
                if (keypoint[2] < keypointThreshold)
                    continue;
                float x = keypoint[1] * inputWidth * scaleX;
                float y = keypoint[0] * inputHeight * scaleY;
                glVertex2f(x, y);
            }
        }
        glEnd();

        glLineWidth(2.0f);
        glBegin(GL_LINES);
        for (int p = 0; p < numPoses; p++) {
            const float* pose = output + (56 * p);
            float score = pose[55];
            if (score < poseThreshold)
                continue;
            int id = poseIds[p];
            Scalar poseColor = g_colors[(id >= 0 ? id : p) % g_colors.size()];
            glColor3f(poseColor[2] / 255.0f, poseColor[1] / 255.0f, poseColor[0] / 255.0f);
            for (const auto &conn : g_connections) {
                const float* kp1 = pose + 3 * conn.first;
                const float* kp2 = pose + 3 * conn.second;
                if (kp1[2] < keypointThreshold || kp2[2] < keypointThreshold)
                    continue;
                float x1 = kp1[1] * inputWidth * scaleX;
                float y1 = kp1[0] * inputHeight * scaleY;
                float x2 = kp2[1] * inputWidth * scaleX;
                float y2 = kp2[0] * inputHeight * scaleY;
                glVertex2f(x1, y1);
                glVertex2f(x2, y2);
            }
        }
        glEnd();
    }

    // Overlay suit parts.
    for (int p = 0; p < numPoses; p++) {
        const float* pose = output + (56 * p);
        float score = pose[55];
        if (score < poseThreshold)
            continue;
        // --- Face overlay using eyes (indices 1 and 2) ---
        const float* nose = pose;  // index 0 is nose
        const float* leftEye = pose + 3 * 1;
        const float* rightEye = pose + 3 * 2;
        if (leftEye[2] >= keypointThreshold && rightEye[2] >= keypointThreshold) {
            float x_left = leftEye[1] * inputWidth * scaleX;
            float y_left = leftEye[0] * inputHeight * scaleY;
            float x_right = rightEye[1] * inputWidth * scaleX;
            float y_right = rightEye[0] * inputHeight * scaleY;
            float faceWidth = norm(Point2f(x_left, y_left) - Point2f(x_right, y_right)) * scales.face;
            float faceHeight = faceWidth * (static_cast<float>(suit.face.height) / suit.face.width);
            float centerX = (x_left + x_right) / 2.0f;
            float centerY = (y_left + y_right) / 2.0f;
            drawTextureRect(centerX, centerY, faceWidth, faceHeight, suit.face);
        } else if (nose[2] >= keypointThreshold) {
            float centerX = nose[1] * inputWidth * scaleX;
            float centerY = nose[0] * inputHeight * scaleY;
            float faceWidth = 80.0f * scales.face;
            float faceHeight = faceWidth * (static_cast<float>(suit.face.height) / suit.face.width);
            drawTextureRect(centerX, centerY, faceWidth, faceHeight, suit.face);
        }

        // --- Hat overlay using ears and nose ---
        const float* leftEar = pose + 3 * 3;
        const float* rightEar = pose + 3 * 4;

        if (leftEar[2] >= keypointThreshold && rightEar[2] >= keypointThreshold) {
            float x_left = leftEar[1] * inputWidth * scaleX;
            float y_left = leftEar[0] * inputHeight * scaleY;
            float x_right = rightEar[1] * inputWidth * scaleX;
            float y_right = rightEar[0] * inputHeight * scaleY;
            float x_nose = nose[1] * inputWidth * scaleX;
            float y_nose = nose[0] * inputHeight * scaleY;
            // Head width based on ear distance
            float headWidth = norm(Point2f(x_left, y_left) - Point2f(x_right, y_right)) * scales.head;
            float headHeight = headWidth * (static_cast<float>(suit.head.height) / suit.head.width);
            // Hat center above nose, slightly above head
            float centerX = x_nose;
            float centerY = y_nose - headHeight * 0.9f;  // Move hat higher
            drawTextureRect(centerX, centerY, headWidth, headHeight, suit.head);
        }

        // --- Torso overlay: from shoulders (indices 5,6) to hips (indices 11,12) ---
        const float* leftShoulder = pose + 3 * 5;
        const float* rightShoulder = pose + 3 * 6;
        const float* leftHip = pose + 3 * 11;
        const float* rightHip = pose + 3 * 12;
        if (leftShoulder[2] >= keypointThreshold && rightShoulder[2] >= keypointThreshold &&
            leftHip[2] >= keypointThreshold && rightHip[2] >= keypointThreshold) {
            float x1 = min({leftShoulder[1], rightShoulder[1], leftHip[1], rightHip[1]}) * inputWidth * scaleX;
            float y1 = min({leftShoulder[0], rightShoulder[0], leftHip[0], rightHip[0]}) * inputHeight * scaleY;
            float x2 = max({leftShoulder[1], rightShoulder[1], leftHip[1], rightHip[1]}) * inputWidth * scaleX;
            float y2 = max({leftShoulder[0], rightShoulder[0], leftHip[0], rightHip[0]}) * inputHeight * scaleY;
            float torsoWidth = (x2 - x1) * scales.torso;
            float torsoHeight = (y2 - y1) * scales.torso;
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            drawTextureRect(centerX, centerY, torsoWidth, torsoHeight, suit.torso);
        }

        // --- Left upper arm: shoulder (index 5) to elbow (index 7) ---
        const float* leftElbow = pose + 3 * 7;
        if (leftShoulder[2] >= keypointThreshold && leftElbow[2] >= keypointThreshold) {
            float x1 = leftShoulder[1] * inputWidth * scaleX;
            float y1 = leftShoulder[0] * inputHeight * scaleY;
            float x2 = leftElbow[1] * inputWidth * scaleX;
            float y2 = leftElbow[0] * inputHeight * scaleY;
            float armLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.leftArm;
            float armHeight = armLength * (static_cast<float>(suit.leftArm.height) / suit.leftArm.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, armLength, armHeight, angle, suit.leftArm);
        }
        // --- Left forearm: elbow (index 7) to wrist (index 9) ---
        const float* leftWrist = pose + 3 * 9;
        if (leftElbow[2] >= keypointThreshold && leftWrist[2] >= keypointThreshold) {
            float x1 = leftElbow[1] * inputWidth * scaleX;
            float y1 = leftElbow[0] * inputHeight * scaleY;
            float x2 = leftWrist[1] * inputWidth * scaleX;
            float y2 = leftWrist[0] * inputHeight * scaleY;
            float forearmLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.leftForearm;
            float forearmHeight = forearmLength * (static_cast<float>(suit.leftForearm.height) / suit.leftForearm.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, forearmLength, forearmHeight, angle, suit.leftForearm);
        }
        // --- Right upper arm: shoulder (index 6) to elbow (index 8) ---
        const float* rightElbow = pose + 3 * 8;
        if (rightShoulder[2] >= keypointThreshold && rightElbow[2] >= keypointThreshold) {
            float x1 = rightShoulder[1] * inputWidth * scaleX;
            float y1 = rightShoulder[0] * inputHeight * scaleY;
            float x2 = rightElbow[1] * inputWidth * scaleX;
            float y2 = rightElbow[0] * inputHeight * scaleY;
            float armLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.rightArm;
            float armHeight = armLength * (static_cast<float>(suit.rightArm.height) / suit.rightArm.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, armLength, armHeight, angle, suit.rightArm);
        }
        // --- Right forearm: elbow (index 8) to wrist (index 10) ---
        const float* rightWrist = pose + 3 * 10;
        if (rightElbow[2] >= keypointThreshold && rightWrist[2] >= keypointThreshold) {
            float x1 = rightElbow[1] * inputWidth * scaleX;
            float y1 = rightElbow[0] * inputHeight * scaleY;
            float x2 = rightWrist[1] * inputWidth * scaleX;
            float y2 = rightWrist[0] * inputHeight * scaleY;
            float forearmLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.rightForearm;
            float forearmHeight = forearmLength * (static_cast<float>(suit.rightForearm.height) / suit.rightForearm.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, forearmLength, forearmHeight, angle, suit.rightForearm);
        }
        // --- Left upper leg: hip (index 11) to knee (index 13) ---
        const float* leftKnee = pose + 3 * 13;
        if (leftHip[2] >= keypointThreshold && leftKnee[2] >= keypointThreshold) {
            float x1 = leftHip[1] * inputWidth * scaleX;
            float y1 = leftHip[0] * inputHeight * scaleY;
            float x2 = leftKnee[1] * inputWidth * scaleX;
            float y2 = leftKnee[0] * inputHeight * scaleY;
            float legLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.leftLeg;
            float legWidth = legLength * (static_cast<float>(suit.leftLeg.height) / suit.leftLeg.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, legLength, legWidth, angle, suit.leftLeg);
        }
        // --- Left lower leg: knee (index 13) to ankle (index 15) ---
        const float* leftAnkle = pose + 3 * 15;
        if (leftKnee[2] >= keypointThreshold && leftAnkle[2] >= keypointThreshold) {
            float x1 = leftKnee[1] * inputWidth * scaleX;
            float y1 = leftKnee[0] * inputHeight * scaleY;
            float x2 = leftAnkle[1] * inputWidth * scaleX;
            float y2 = leftAnkle[0] * inputHeight * scaleY;
            float lowerLegLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.leftLowerLeg;
            float lowerLegWidth = lowerLegLength * (static_cast<float>(suit.leftLowerLeg.height) / suit.leftLowerLeg.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, lowerLegLength, lowerLegWidth, angle, suit.leftLowerLeg);
        }
        // --- Right upper leg: hip (index 12) to knee (index 14) ---
        const float* rightKnee = pose + 3 * 14;
        if (rightHip[2] >= keypointThreshold && rightKnee[2] >= keypointThreshold) {
            float x1 = rightHip[1] * inputWidth * scaleX;
            float y1 = rightHip[0] * inputHeight * scaleY;
            float x2 = rightKnee[1] * inputWidth * scaleX;
            float y2 = rightKnee[0] * inputHeight * scaleY;
            float legLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.rightLeg;
            float legWidth = legLength * (static_cast<float>(suit.rightLeg.height) / suit.rightLeg.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, legLength, legWidth, angle, suit.rightLeg);
        }
        // --- Right lower leg: knee (index 14) to ankle (index 16) ---
        const float* rightAnkle = pose + 3 * 16;
        if (rightKnee[2] >= keypointThreshold && rightAnkle[2] >= keypointThreshold) {
            float x1 = rightKnee[1] * inputWidth * scaleX;
            float y1 = rightKnee[0] * inputHeight * scaleY;
            float x2 = rightAnkle[1] * inputWidth * scaleX;
            float y2 = rightAnkle[0] * inputHeight * scaleY;
            float lowerLegLength = norm(Point2f(x2, y2) - Point2f(x1, y1)) * scales.rightLowerLeg;
            float lowerLegWidth = lowerLegLength * (static_cast<float>(suit.rightLowerLeg.height) / suit.rightLowerLeg.width);
            float centerX = (x1 + x2) / 2.0f;
            float centerY = (y1 + y2) / 2.0f;
            float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
            drawRotatedTextureRect(centerX, centerY, lowerLegLength, lowerLegWidth, angle, suit.rightLowerLeg);
        }
    }
}
