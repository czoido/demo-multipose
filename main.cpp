// main_multipose.cpp
// Multipose tracking example that uses persistent pose tracking.
// - Captures full-resolution frames from a video input (webcam or file).
// - Creates a small inference image (192x192) for TFLite multipose inference.
// - Uses the full-resolution frame (resized to window size) for display.
// - Draws pose keypoints and connections (persistent colors) over the display image.
// - The input source is specified as a command-line argument (default is "0" for webcam).
// - The SDL window is sized based on the input resolution but limited to the desktop size.
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

using namespace std;
using namespace cv;

// Global constants for pose drawing.
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

//-------------------------
// PoseTracker: assigns persistent IDs to poses.
//-------------------------
struct PoseData {
    int id;
    Point2f center;
};

class PoseTracker {
public:
    PoseTracker() : nextId(0) {}

    // Given the output tensor data, number of poses, input resolution,
    // and thresholds, compute each valid pose's center and assign persistent IDs.
    vector<int> trackPoses(const float* output, int numPoses, int inpWidth, int inpHeight,
                           float poseThreshold, float keypointThreshold) {
        vector<PoseData> currentPoses;
        vector<int> poseIndices; // Stores indices of valid poses.

        // Extract centers for each pose.
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
                poseIndices.push_back(p);
            }
        }

        // Match current poses to previous poses using nearest-neighbor distance.
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
            // If a previous pose is found within a threshold, assign its id.
            if (bestIdx != -1 && bestDist < 50.0f) {
                curr.id = prevPoses[bestIdx].id;
                used[bestIdx] = true;
            } else {
                curr.id = nextId++;
            }
        }
        prevPoses = currentPoses;

        // Build result vector: map each original pose index to its persistent id.
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

//-------------------------
// VideoInput: wraps OpenCV VideoCapture.
//-------------------------
class VideoInput {
public:
    // The source parameter can be either a camera index (as a string) or a file path.
    VideoInput(const string &source) {
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
    // Get a full-resolution frame.
    bool getFrame(Mat &frame) {
        cap >> frame;
        if (frame.empty() && !isCamera) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }
        return !frame.empty();
    }
    double getFrameWidth() const {
        return cap.get(CAP_PROP_FRAME_WIDTH);
    }
    double getFrameHeight() const {
        return cap.get(CAP_PROP_FRAME_HEIGHT);
    }
private:
    VideoCapture cap;
    bool isCamera;
};

//-------------------------
// PoseEstimator: loads and runs the TFLite multipose model and draws pose overlays.
//-------------------------
class PoseEstimator {
public:
    // modelPath: path to the TFLite model.
    // multiPose: whether to use multipose.
    // inpWidth/inpHeight: inference resolution.
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
    // Run inference on a small (inpWidth x inpHeight) image.
    float* runInference(const Mat &inferenceImage) {
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), inferenceImage.data,
               inferenceImage.total() * inferenceImage.elemSize());
        if (interpreter->Invoke() != kTfLiteOk)
            cerr << "Inference failed" << endl;
        return interpreter->typed_output_tensor<float>(0);
    }
    // Draw pose keypoints and connections using OpenGL.
    // displayImage: full-resolution image (resized to window size).
    // The keypoints output is relative to the inference resolution.
    void drawPosesGL(const Mat &displayImage, float* output) {
        int dispWidth = displayImage.cols;
        int dispHeight = displayImage.rows;
        float scaleX = static_cast<float>(dispWidth) / static_cast<float>(inputWidth);
        float scaleY = static_cast<float>(dispHeight) / static_cast<float>(inputHeight);
        int numPoses = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];

        // Obtain persistent pose IDs.
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

//-------------------------
// OpenGLRenderer: handles SDL2 window and OpenGL texture rendering.
//-------------------------
class OpenGLRenderer {
public:
    OpenGLRenderer(int initialWidth, int initialHeight)
            : windowWidth(initialWidth), windowHeight(initialHeight),
              textureWidth(initialWidth), textureHeight(initialHeight)
    {
        window = SDL_CreateWindow("Multipose Tracking", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
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
        // Allocate texture storage using current window size.
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
        glColor3f(1.0f, 1.0f, 1.0f); // Set color to white.
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

//-------------------------
// Main function.
//-------------------------
int main(int argc, char* argv[]) {
    try {
        // Initialize SDL video subsystem.
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw runtime_error(string("SDL_Init Error: ") + SDL_GetError());
        }

        // Default source is "0" (webcam device 0); can be overridden via command-line.
        string source = "0";
        if (argc > 1) {
            source = argv[1];
        }
        VideoInput videoInput(source);

        // Get input resolution.
        int inputW = static_cast<int>(videoInput.getFrameWidth());
        int inputH = static_cast<int>(videoInput.getFrameHeight());
        cout << "Input resolution: " << inputW << "x" << inputH << endl;

        // Query desktop resolution.
        SDL_DisplayMode dm;
        if (SDL_GetCurrentDisplayMode(0, &dm) != 0) {
            throw runtime_error(string("SDL_GetCurrentDisplayMode Error: ") + SDL_GetError());
        }
        int desktopW = dm.w;
        int desktopH = dm.h;
        cout << "Desktop resolution: " << desktopW << "x" << desktopH << endl;

        // Compute scaling factor to limit window size to desktop dimensions.
        float scaleFactor = min(1.0f, min(static_cast<float>(desktopW) / inputW, static_cast<float>(desktopH) / inputH));
        int windowW = static_cast<int>(inputW * scaleFactor);
        int windowH = static_cast<int>(inputH * scaleFactor);
        cout << "Window resolution: " << windowW << "x" << windowH << endl;

        // Create PoseEstimator using a small inference size (192x192).
        string modelPath = "../lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";
        PoseEstimator poseEstimator(modelPath, true, 192, 192);

        // Create OpenGLRenderer with the computed window resolution.
        OpenGLRenderer renderer(windowW, windowH);

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

            // Resize for inference.
            resize(fullFrame, inferenceFrame, Size(192, 192));
            float* output = poseEstimator.runInference(inferenceFrame);

            // For display: resize full frame to current window size.
            int winWCurrent = renderer.getWidth();
            int winHCurrent = renderer.getHeight();
            resize(fullFrame, dispFrame, Size(winWCurrent, winHCurrent));
            // Convert BGR to RGB.
            cvtColor(dispFrame, rgbDispFrame, COLOR_BGR2RGB);
            if (!rgbDispFrame.isContinuous())
                rgbDispFrame = rgbDispFrame.clone();

            renderer.updateTexture(rgbDispFrame);
            glClear(GL_COLOR_BUFFER_BIT);
            renderer.renderQuad();
            poseEstimator.drawPosesGL(dispFrame, output);
            SDL_GL_SwapWindow(renderer.getWindow());
        }
    }
    catch (const exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }
    return 0;
}
