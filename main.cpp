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

//------------------------------------------------------------
// VideoInput class: wraps OpenCV VideoCapture for generic input.
//------------------------------------------------------------
class VideoInput {
public:
    // The source parameter can be either a camera index (as string) or a file path.
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
    // Get input frame width.
    double getFrameWidth() const {
        return cap.get(CAP_PROP_FRAME_WIDTH);
    }
    // Get input frame height.
    double getFrameHeight() const {
        return cap.get(CAP_PROP_FRAME_HEIGHT);
    }
private:
    VideoCapture cap;
    bool isCamera;
};

//------------------------------------------------------------
// PoseEstimator class: loads and runs the TFLite multipose model
// and draws pose overlays scaled to a given display resolution.
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
    void drawPosesGL(const Mat &displayImage, float* output) {
        int dispWidth = displayImage.cols;
        int dispHeight = displayImage.rows;
        float scaleX = static_cast<float>(dispWidth) / static_cast<float>(inputWidth);
        float scaleY = static_cast<float>(dispHeight) / static_cast<float>(inputHeight);
        int numPoses = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];

        glPointSize(8.0f);
        glBegin(GL_POINTS);
        for (int p = 0; p < numPoses; p++) {
            float* pose = output + (56 * p);
            float score = pose[55];
            if (score < poseThreshold)
                continue;
            Scalar poseColor = g_colors[p % g_colors.size()];
            glColor3f(poseColor[2] / 255.0f, poseColor[1] / 255.0f, poseColor[0] / 255.0f);
            for (int k = 0; k < 17; k++) {
                float* keypoint = pose + 3 * k;
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
            float* pose = output + (56 * p);
            float score = pose[55];
            if (score < poseThreshold)
                continue;
            Scalar poseColor = g_colors[p % g_colors.size()];
            glColor3f(poseColor[2] / 255.0f, poseColor[1] / 255.0f, poseColor[0] / 255.0f);
            for (const auto &conn : g_connections) {
                float* kp1 = pose + 3 * conn.first;
                float* kp2 = pose + 3 * conn.second;
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
};

//------------------------------------------------------------
// OpenGLRenderer class: handles SDL2 window and OpenGL texture rendering.
//------------------------------------------------------------
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
        glColor3f(1.0f, 1.0f, 1.0f); // White color for texture.
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
// Main function: integrates full-resolution capture,
// inference on a small image, and scaled display with pose overlay.
// The input source is selected via a command-line argument.
// The SDL window is initialized to the input resolution, limited
// by the desktop resolution.
//------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        // Initialize SDL video subsystem.
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw runtime_error(string("SDL_Init Error: ") + SDL_GetError());
        }

        // Default source is "0" (webcam device 0).
        string source = "0";
        if (argc > 1) {
            source = argv[1];
        }

        // Create video input instance.
        VideoInput videoInput(source);

        // Obtain input resolution.
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

        // Calculate scaling factor to limit window size.
        float scaleFactor = min(1.0f, min(static_cast<float>(desktopW) / inputW, static_cast<float>(desktopH) / inputH));
        int windowW = static_cast<int>(inputW * scaleFactor);
        int windowH = static_cast<int>(inputH * scaleFactor);
        cout << "Window resolution: " << windowW << "x" << windowH << endl;

        // Create PoseEstimator.
        string modelPath = "../lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";
        PoseEstimator poseEstimator(modelPath, true, 192, 192);

        // Create OpenGLRenderer with computed window resolution.
        OpenGLRenderer renderer(windowW, windowH);

        bool running = true;
        SDL_Event event;
        Mat fullFrame;
        Mat inferenceFrame;
        Mat dispFrame;
        Mat rgbDispFrame;

        while (running) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    running = false;
                else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED)
                    renderer.updateViewport();
            }

            if (!videoInput.getFrame(fullFrame))
                continue;
            flip(fullFrame, fullFrame, 1);

            resize(fullFrame, inferenceFrame, Size(192, 192));
            float* output = poseEstimator.runInference(inferenceFrame);

            int winWCurrent = renderer.getWidth();
            int winHCurrent = renderer.getHeight();
            resize(fullFrame, dispFrame, Size(winWCurrent, winHCurrent));
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
