// main_multipose_suit_complete.cpp
// Multipose tracking with complete suit overlay.
// - Captures full‑resolution frames from a video source (webcam or file).
// - Resizes a copy to 192x192 for TFLite multipose inference.
// - Displays the full‑resolution frame (resized to window size) using SDL2 and OpenGL.
// - Draws persistent pose keypoints and connections.
// - Overlays separate suit‑part textures (face, torso, arms, forearms, legs) on each detected person.
// - The input source is specified via a command‑line argument (default "0" for webcam).
// - The SDL window is sized based on the input resolution but limited to the desktop dimensions.
// All comments are in English.

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <iostream>
#include <map>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;
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

const float poseThreshold = 0.2f;
const float keypointThreshold = 0.2f;

//------------------------------------------------------------
// PoseTracker: assigns persistent IDs to detected poses.
//------------------------------------------------------------
struct PoseData {
    int id;
    Point2f center;
};

class PoseTracker {
public:
    PoseTracker() : nextId(0) {}

    // For each valid pose, compute the center from all keypoints and then use nearest-neighbor
    // matching with the previous frame to assign a persistent ID.
    vector<int> trackPoses(const float* output, int numPoses, int inpWidth, int inpHeight,
                           float poseThreshold, float keypointThreshold) {
        vector<PoseData> currentPoses;
        for (int p = 0; p < numPoses; p++) {
            const float* pose = output + (56 * p);
            float score = pose[55];
            if (score < poseThreshold)
                continue;
            Point2f center(0, 0);
            int count = 0;
            for (int k = 0; k < 17; k++) {
                const float* kp = pose + 3 * k;
                if (kp[2] < keypointThreshold)
                    continue;
                center.x += kp[1] * inpWidth;
                center.y += kp[0] * inpHeight;
                count++;
            }
            if (count > 0) {
                center.x /= count;
                center.y /= count;
                currentPoses.push_back({-1, center});
            }
        }

        vector<bool> used(prevPoses.size(), false);
        for (auto &curr : currentPoses) {
            float bestDist = numeric_limits<float>::max();
            int bestIdx = -1;
            for (int i = 0; i < prevPoses.size(); i++) {
                if (used[i])
                    continue;
                float dist = norm(curr.center - prevPoses[i].center);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            if (bestIdx != -1 && bestDist < 50.0f) {
                curr.id = prevPoses[bestIdx].id;
                used[bestIdx] = true;
            } else {
                curr.id = nextId++;
            }
        }
        prevPoses = currentPoses;
        vector<int> result(numPoses, -1);
        int j = 0;
        for (int p = 0; p < numPoses; p++) {
            const float* pose = output + (56 * p);
            if (pose[55] < poseThreshold)
                continue;
            result[p] = currentPoses[j].id;
            j++;
        }
        return result;
    }
private:
    vector<PoseData> prevPoses;
    int nextId;
};

//------------------------------------------------------------
// MaskTexture: holds an OpenGL texture ID and its original dimensions.
//------------------------------------------------------------
struct MaskTexture {
    GLuint textureID;
    int width;
    int height;
};

//------------------------------------------------------------
// loadMaskTexture: loads an image file (with alpha) into an OpenGL texture.
//------------------------------------------------------------
MaskTexture loadMaskTexture(const string& filename) {
    Mat img = imread(filename, IMREAD_UNCHANGED);
    if (img.empty())
        throw runtime_error("Failed to load mask image: " + filename);
    if (img.channels() == 4)
        cvtColor(img, img, COLOR_BGRA2RGBA);
    MaskTexture mask;
    glGenTextures(1, &mask.textureID);
    glBindTexture(GL_TEXTURE_2D, mask.textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
    mask.width = img.cols;
    mask.height = img.rows;
    return mask;
}

//------------------------------------------------------------
// drawTextureRect: draws a textured quad without rotation.
//------------------------------------------------------------
void drawTextureRect(float centerX, float centerY, float width, float height, const MaskTexture &tex) {
    float halfW = width / 2.0f;
    float halfH = height / 2.0f;
    glColor3f(1.0f, 1.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, tex.textureID);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(centerX - halfW, centerY - halfH);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(centerX + halfW, centerY - halfH);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(centerX + halfW, centerY + halfH);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(centerX - halfW, centerY + halfH);
    glEnd();
    glDisable(GL_BLEND);
}

//------------------------------------------------------------
// drawRotatedTextureRect: draws a textured quad rotated by angle (in degrees).
//------------------------------------------------------------
void drawRotatedTextureRect(float centerX, float centerY, float width, float height, float angle, const MaskTexture &tex) {
    float halfW = width / 2.0f;
    float halfH = height / 2.0f;
    glPushMatrix();
    glTranslatef(centerX, centerY, 0);
    glRotatef(angle, 0, 0, 1);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, tex.textureID);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-halfW, -halfH);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(halfW, -halfH);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(halfW, halfH);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-halfW, halfH);
    glEnd();
    glDisable(GL_BLEND);
    glPopMatrix();
}

//------------------------------------------------------------
// SuitTextures: holds textures for various suit parts.
//------------------------------------------------------------
struct SuitTextures {
    MaskTexture face;
    MaskTexture torso;
    MaskTexture leftArm;
    MaskTexture rightArm;
    MaskTexture leftForearm;
    MaskTexture rightForearm;
    MaskTexture leftLeg;
    MaskTexture rightLeg;
    MaskTexture leftLowerLeg;
    MaskTexture rightLowerLeg;
};

//------------------------------------------------------------
// VideoInputWrapper: wraps OpenCV VideoCapture for webcam or file input.
//------------------------------------------------------------
class VideoInputWrapper {
public:
    VideoInputWrapper(const string &source) {
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
    bool getFrame(Mat &frame) {
        cap >> frame;
        if (frame.empty() && !isCamera) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }
        return !frame.empty();
    }
    double getFrameWidth() const { return cap.get(CAP_PROP_FRAME_WIDTH); }
    double getFrameHeight() const { return cap.get(CAP_PROP_FRAME_HEIGHT); }
private:
    VideoCapture cap;
    bool isCamera;
};

//------------------------------------------------------------
// PoseEstimator: loads and runs the TFLite multipose model and draws pose overlays and suit textures.
//------------------------------------------------------------
class PoseEstimator {
public:
    PoseEstimator(const string &modelPath, bool multiPose = true, int inpWidth = 192, int inpHeight = 192)
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
    float* runInference(const Mat &inferenceImage) {
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), inferenceImage.data,
               inferenceImage.total() * inferenceImage.elemSize());
        if (interpreter->Invoke() != kTfLiteOk)
            cerr << "Inference failed" << endl;
        return interpreter->typed_output_tensor<float>(0);
    }
    // Draws keypoints, connections, and overlays suit textures.
    // displayImage: full‑resolution image (resized to window size).
    // suit: SuitTextures structure with suit part textures.
    void drawPosesGL(const Mat &displayImage, float* output, const SuitTextures &suit) {
        int dispWidth = displayImage.cols;
        int dispHeight = displayImage.rows;
        float scaleX = static_cast<float>(dispWidth) / static_cast<float>(inputWidth);
        float scaleY = static_cast<float>(dispHeight) / static_cast<float>(inputHeight);
        int numPoses = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];

        // Get persistent pose IDs.
        vector<int> poseIds = tracker.trackPoses(output, numPoses, inputWidth, inputHeight, poseThreshold, keypointThreshold);

        // Draw keypoints.
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

        // Draw connections.
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

        // For each pose, overlay suit parts.
        for (int p = 0; p < numPoses; p++) {
            const float* pose = output + (56 * p);
            float score = pose[55];
            if (score < poseThreshold)
                continue;

            // --- Face overlay ---
            const float* nose = pose; // index 0
            const float* leftEar = pose + 3 * 3;   // index 3
            const float* rightEar = pose + 3 * 4;  // index 4
            if (leftEar[2] >= keypointThreshold && rightEar[2] >= keypointThreshold) {
                float x_left = leftEar[1] * inputWidth * scaleX;
                float y_left = leftEar[0] * inputHeight * scaleY;
                float x_right = rightEar[1] * inputWidth * scaleX;
                float y_right = rightEar[0] * inputHeight * scaleY;
                float faceWidth = norm(Point2f(x_left, y_left) - Point2f(x_right, y_right));
                float faceHeight = faceWidth * (suit.face.height / static_cast<float>(suit.face.width));
                float centerX = (x_left + x_right) / 2.0f;
                float centerY = (y_left + y_right) / 2.0f;
                drawTextureRect(centerX, centerY, faceWidth, faceHeight, suit.face);
            } else if (nose[2] >= keypointThreshold) {
                float centerX = nose[1] * inputWidth * scaleX;
                float centerY = nose[0] * inputHeight * scaleY;
                float faceWidth = 80.0f;
                float faceHeight = faceWidth * (suit.face.height / static_cast<float>(suit.face.width));
                drawTextureRect(centerX, centerY, faceWidth, faceHeight, suit.face);
            }

            // --- Torso overlay: from shoulders (5,6) to hips (11,12) ---
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
                float torsoWidth = x2 - x1;
                float torsoHeight = y2 - y1;
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                drawTextureRect(centerX, centerY, torsoWidth, torsoHeight, suit.torso);
            }

            // --- Left upper arm: shoulder (5) to elbow (7) ---
            const float* leftElbow = pose + 3 * 7;
            if (leftShoulder[2] >= keypointThreshold && leftElbow[2] >= keypointThreshold) {
                float x1 = leftShoulder[1] * inputWidth * scaleX;
                float y1 = leftShoulder[0] * inputHeight * scaleY;
                float x2 = leftElbow[1] * inputWidth * scaleX;
                float y2 = leftElbow[0] * inputHeight * scaleY;
                float armLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float armHeight = armLength * (suit.leftArm.height / static_cast<float>(suit.leftArm.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, armLength, armHeight, angle, suit.leftArm);
            }
            // --- Left forearm: elbow (7) to wrist (9) ---
            const float* leftWrist = pose + 3 * 9;
            if (leftElbow[2] >= keypointThreshold && leftWrist[2] >= keypointThreshold) {
                float x1 = leftElbow[1] * inputWidth * scaleX;
                float y1 = leftElbow[0] * inputHeight * scaleY;
                float x2 = leftWrist[1] * inputWidth * scaleX;
                float y2 = leftWrist[0] * inputHeight * scaleY;
                float forearmLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float forearmHeight = forearmLength * (suit.leftForearm.height / static_cast<float>(suit.leftForearm.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, forearmLength, forearmHeight, angle, suit.leftForearm);
            }
            // --- Right upper arm: shoulder (6) to elbow (8) ---
            const float* rightElbow = pose + 3 * 8;
            if (rightShoulder[2] >= keypointThreshold && rightElbow[2] >= keypointThreshold) {
                float x1 = rightShoulder[1] * inputWidth * scaleX;
                float y1 = rightShoulder[0] * inputHeight * scaleY;
                float x2 = rightElbow[1] * inputWidth * scaleX;
                float y2 = rightElbow[0] * inputHeight * scaleY;
                float armLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float armHeight = armLength * (suit.rightArm.height / static_cast<float>(suit.rightArm.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, armLength, armHeight, angle, suit.rightArm);
            }
            // --- Right forearm: elbow (8) to wrist (10) ---
            const float* rightWrist = pose + 3 * 10;
            if (rightElbow[2] >= keypointThreshold && rightWrist[2] >= keypointThreshold) {
                float x1 = rightElbow[1] * inputWidth * scaleX;
                float y1 = rightElbow[0] * inputHeight * scaleY;
                float x2 = rightWrist[1] * inputWidth * scaleX;
                float y2 = rightWrist[0] * inputHeight * scaleY;
                float forearmLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float forearmHeight = forearmLength * (suit.rightForearm.height / static_cast<float>(suit.rightForearm.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, forearmLength, forearmHeight, angle, suit.rightForearm);
            }
            // --- Left upper leg: hip (11) to knee (13) ---
            const float* leftKnee = pose + 3 * 13;
            if (leftHip[2] >= keypointThreshold && leftKnee[2] >= keypointThreshold) {
                float x1 = leftHip[1] * inputWidth * scaleX;
                float y1 = leftHip[0] * inputHeight * scaleY;
                float x2 = leftKnee[1] * inputWidth * scaleX;
                float y2 = leftKnee[0] * inputHeight * scaleY;
                float legLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float legWidth = legLength * (suit.leftLeg.height / static_cast<float>(suit.leftLeg.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, legLength, legWidth, angle, suit.leftLeg);
            }
            // --- Left lower leg: knee (13) to ankle (15) ---
            const float* leftAnkle = pose + 3 * 15;
            if (leftKnee[2] >= keypointThreshold && leftAnkle[2] >= keypointThreshold) {
                float x1 = leftKnee[1] * inputWidth * scaleX;
                float y1 = leftKnee[0] * inputHeight * scaleY;
                float x2 = leftAnkle[1] * inputWidth * scaleX;
                float y2 = leftAnkle[0] * inputHeight * scaleY;
                float lowerLegLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float lowerLegWidth = lowerLegLength * (suit.leftLowerLeg.height / static_cast<float>(suit.leftLowerLeg.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, lowerLegLength, lowerLegWidth, angle, suit.leftLowerLeg);
            }
            // --- Right upper leg: hip (12) to knee (14) ---
            const float* rightKnee = pose + 3 * 14;
            if (rightHip[2] >= keypointThreshold && rightKnee[2] >= keypointThreshold) {
                float x1 = rightHip[1] * inputWidth * scaleX;
                float y1 = rightHip[0] * inputHeight * scaleY;
                float x2 = rightKnee[1] * inputWidth * scaleX;
                float y2 = rightKnee[0] * inputHeight * scaleY;
                float legLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float legWidth = legLength * (suit.rightLeg.height / static_cast<float>(suit.rightLeg.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, legLength, legWidth, angle, suit.rightLeg);
            }
            // --- Right lower leg: knee (14) to ankle (16) ---
            const float* rightAnkle = pose + 3 * 16;
            if (rightKnee[2] >= keypointThreshold && rightAnkle[2] >= keypointThreshold) {
                float x1 = rightKnee[1] * inputWidth * scaleX;
                float y1 = rightKnee[0] * inputHeight * scaleY;
                float x2 = rightAnkle[1] * inputWidth * scaleX;
                float y2 = rightAnkle[0] * inputHeight * scaleY;
                float lowerLegLength = norm(Point2f(x2, y2) - Point2f(x1, y1));
                float lowerLegWidth = lowerLegLength * (suit.rightLowerLeg.height / static_cast<float>(suit.rightLowerLeg.width));
                float centerX = (x1 + x2) / 2.0f;
                float centerY = (y1 + y2) / 2.0f;
                float angle = atan2(y2 - y1, x2 - x1) * 180.0f / CV_PI;
                drawRotatedTextureRect(centerX, centerY, lowerLegLength, lowerLegWidth, angle, suit.rightLowerLeg);
            }
        }
    }
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

//------------------------------------------------------------
// OpenGLRenderer: manages the SDL window and texture rendering with OpenGL.
//------------------------------------------------------------
class OpenGLRenderer {
public:
    OpenGLRenderer(int initialWidth, int initialHeight)
            : windowWidth(initialWidth), windowHeight(initialHeight),
              textureWidth(initialWidth), textureHeight(initialHeight)
    {
        window = SDL_CreateWindow("Multipose Suit Overlay", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  windowWidth, windowHeight,
                                  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
        if (!window)
            throw runtime_error(string("SDL_CreateWindow Error: ") + SDL_GetError());
        glContext = SDL_GL_CreateContext(window);
        if (!glContext) {
            SDL_DestroyWindow(window);
            throw runtime_error(string("SDL_GL_CreateContext Error: ") + SDL_GetError());
        }
        updateViewport();
        glEnable(GL_TEXTURE_2D);
        glClearColor(0, 0, 0, 1);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    }
    ~OpenGLRenderer() {
        glDeleteTextures(1, &textureID);
        SDL_GL_DeleteContext(glContext);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    void updateViewport() {
        SDL_GetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
    void updateTexture(const Mat &frame) {
        glBindTexture(GL_TEXTURE_2D, textureID);
        if (frame.cols != textureWidth || frame.rows != textureHeight) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
            textureWidth = frame.cols;
            textureHeight = frame.rows;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
        }
        glFlush();
    }
    void renderQuad() {
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(windowWidth, 0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(windowWidth, windowHeight);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0, windowHeight);
        glEnd();
    }
    SDL_Window* getWindow() { return window; }
    int getWidth() const { return windowWidth; }
    int getHeight() const { return windowHeight; }
private:
    SDL_Window* window;
    SDL_GLContext glContext;
    GLuint textureID;
    int windowWidth;
    int windowHeight;
    int textureWidth;
    int textureHeight;
};

//------------------------------------------------------------
// Main: sets up video input, TFLite inference, and OpenGL rendering with suit overlay.
//------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        // Set SDL OpenGL attributes.
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

        // Initialize SDL video subsystem.
        if (SDL_Init(SDL_INIT_VIDEO) < 0)
            throw runtime_error(string("SDL_Init Error: ") + SDL_GetError());

        // Default source is "0" (webcam); can be overridden via command-line.
        string source = "0";
        if (argc > 1)
            source = argv[1];
        VideoInputWrapper videoInput(source);

        // Get input resolution.
        int inputW = static_cast<int>(videoInput.getFrameWidth());
        int inputH = static_cast<int>(videoInput.getFrameHeight());
        cout << "Input resolution: " << inputW << "x" << inputH << endl;

        // Query desktop resolution.
        SDL_DisplayMode dm;
        if (SDL_GetCurrentDisplayMode(0, &dm) != 0)
            throw runtime_error(string("SDL_GetCurrentDisplayMode Error: ") + SDL_GetError());
        int desktopW = dm.w;
        int desktopH = dm.h;
        cout << "Desktop resolution: " << desktopW << "x" << desktopH << endl;

        // Compute scale factor to limit window size while preserving aspect ratio.
        float scaleFactor = min(1.0f, min(static_cast<float>(desktopW) / inputW, static_cast<float>(desktopH) / inputH));
        int windowW = static_cast<int>(inputW * scaleFactor);
        int windowH = static_cast<int>(inputH * scaleFactor);
        cout << "Window resolution: " << windowW << "x" << windowH << endl;

        // Create OpenGLRenderer.
        OpenGLRenderer renderer(windowW, windowH);

        // Now that the GL context is active, load suit textures.
        SuitTextures suit;
        suit.face = loadMaskTexture("../suit_face.png");
        suit.torso = loadMaskTexture("../suit_torso.png");
        suit.leftArm = loadMaskTexture("../suit_left_arm.png");
        suit.rightArm = loadMaskTexture("../suit_right_arm.png");
        suit.leftForearm = loadMaskTexture("../suit_left_forearm.png");
        suit.rightForearm = loadMaskTexture("../suit_right_forearm.png");
        suit.leftLeg = loadMaskTexture("../suit_left_leg.png");
        suit.rightLeg = loadMaskTexture("../suit_right_leg.png");
        suit.leftLowerLeg = loadMaskTexture("../suit_left_lower_leg.png");
        suit.rightLowerLeg = loadMaskTexture("../suit_right_lower_leg.png");

        // Create PoseEstimator with inference resolution 192x192.
        string modelPath = "../lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";
        PoseEstimator poseEstimator(modelPath, true, 192, 192);

        bool running = true;
        SDL_Event event;
        Mat fullFrame, inferenceFrame, dispFrame, rgbDispFrame;

        while (running) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    running = false;
                else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED)
                    renderer.updateViewport();
            }
            if (!videoInput.getFrame(fullFrame))
                continue;
            // Mirror frame horizontally.
            flip(fullFrame, fullFrame, 1);

            // Resize a copy for inference.
            resize(fullFrame, inferenceFrame, Size(192, 192));
            float* output = poseEstimator.runInference(inferenceFrame);

            // For display: resize full frame to current window size.
            int winWCurrent = renderer.getWidth();
            int winHCurrent = renderer.getHeight();
            resize(fullFrame, dispFrame, Size(winWCurrent, winHCurrent));
            cvtColor(dispFrame, rgbDispFrame, COLOR_BGR2RGB);
            if (!rgbDispFrame.isContinuous())
                rgbDispFrame = rgbDispFrame.clone();

            renderer.updateTexture(rgbDispFrame);
            glClear(GL_COLOR_BUFFER_BIT);
            renderer.renderQuad();
            // Draw pose overlays and suit overlays.
            poseEstimator.drawPosesGL(dispFrame, output, suit);
            SDL_GL_SwapWindow(renderer.getWindow());
        }
    }
    catch (const exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }
    return 0;
}
