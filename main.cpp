// main_multipose_refactored.cpp
// Refactored multipose tracking example.
// - Captures full-resolution frames from the webcam.
// - Creates a small inference image (192x192) to run TFLite multipose inference.
// - Uses the full-resolution frame (resized to window size) for display.
// - Draws pose keypoints and connections over the display image,
//   scaling the keypoints from the inference resolution to display resolution.
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

using namespace std;
using namespace cv;

// Global constants used for pose drawing.
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
// WebcamCapture class: wraps OpenCV VideoCapture.
//------------------------------------------------------------
class WebcamCapture {
public:
    WebcamCapture(int device = 0) {
        cap.open(device);
        if (!cap.isOpened())
            throw runtime_error("Failed to open webcam");
    }
    // Returns a full-resolution frame.
    bool getFrame(Mat &frame) {
        cap >> frame;
        return !frame.empty();
    }
private:
    VideoCapture cap;
};

//------------------------------------------------------------
// PoseEstimator class: loads and runs the TFLite multipose model
// and draws pose overlays scaled to a given display resolution.
//------------------------------------------------------------
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

    // Run inference on a small (192x192) image.
    float* runInference(const Mat &inferenceImage) {
        // Assumes inferenceImage is already of size inputWidth x inputHeight.
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), inferenceImage.data,
               inferenceImage.total() * inferenceImage.elemSize());
        if (interpreter->Invoke() != kTfLiteOk)
            cerr << "Inference failed" << endl;
        return interpreter->typed_output_tensor<float>(0);
    }

    // Draw pose keypoints and connections using OpenGL.
    // displayImage: full-resolution image (already resized to window size).
    // The keypoints output is relative to the inference resolution.
    void drawPosesGL(const Mat &displayImage, float* output) {
        int dispWidth = displayImage.cols;
        int dispHeight = displayImage.rows;
        // Scaling factors from inference resolution to display resolution.
        float scaleX = static_cast<float>(dispWidth) / static_cast<float>(inputWidth);
        float scaleY = static_cast<float>(dispHeight) / static_cast<float>(inputHeight);

        // Get number of poses from the output tensor shape.
        int numPoses = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];

        // Draw keypoints.
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

        // Draw connections.
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
        // Create SDL window with OpenGL context (resizable)
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
        // Create texture; initial allocation with window dimensions.
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

    // Update viewport and projection on window resize.
    void updateViewport() {
        SDL_GetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    // Update the texture with the given frame (in RGB).
    // This function checks against the stored texture dimensions.
    void updateTexture(const Mat &frame) {
        glBindTexture(GL_TEXTURE_2D, textureID);
        if (frame.cols != textureWidth || frame.rows != textureHeight) {
            // Reallocate texture storage and update stored dimensions.
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
            textureWidth = frame.cols;
            textureHeight = frame.rows;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
        }
        glFlush();
    }

    // Render a textured quad covering the entire window.
    void renderQuad() {
        glColor3f(1.0,1.0,1.0);
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
    int textureWidth;  // Currently allocated texture width.
    int textureHeight; // Currently allocated texture height.
};

//------------------------------------------------------------
// Main function: integrates full-resolution capture,
// inference on a small image, and scaled display with pose overlay.
//------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        // Create webcam capture instance.
        WebcamCapture webcam(0);

        // Create PoseEstimator using a small inference size (192x192).
        string modelPath = "../lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";
        PoseEstimator poseEstimator(modelPath, true, 192, 192);

        // Create OpenGLRenderer with a default display resolution (e.g., 640x480).
        int displayDefaultWidth = 640;
        int displayDefaultHeight = 480;
        OpenGLRenderer renderer(displayDefaultWidth, displayDefaultHeight);

        bool running = true;
        SDL_Event event;
        Mat fullFrame;         // Full-resolution frame.
        Mat inferenceFrame;    // Resized frame for inference (192x192).
        Mat dispFrame;         // Full-resolution frame resized to window size.
        Mat rgbDispFrame;      // dispFrame converted to RGB.

        auto lastTime = chrono::steady_clock::now();

        while (running) {
            // Process SDL events.
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    running = false;
                else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED)
                    renderer.updateViewport();
            }

            // Capture a full-resolution frame.
            if (!webcam.getFrame(fullFrame))
                continue;
            // Mirror full frame horizontally.
            flip(fullFrame, fullFrame, 1);

            // Create inference frame: resize fullFrame to 192x192.
            resize(fullFrame, inferenceFrame, Size(192, 192));

            // Run multipose inference.
            float* output = poseEstimator.runInference(inferenceFrame);

            // For display, resize the fullFrame to the current window size.
            int winW = renderer.getWidth();
            int winH = renderer.getHeight();
            resize(fullFrame, dispFrame, Size(winW, winH));
            // Convert display frame from BGR to RGB.
            cvtColor(dispFrame, rgbDispFrame, COLOR_BGR2RGB);
            if (!rgbDispFrame.isContinuous())
                rgbDispFrame = rgbDispFrame.clone();

            // Update the texture with the display image.
            renderer.updateTexture(rgbDispFrame);

            // Clear the screen.
            glClear(GL_COLOR_BUFFER_BIT);

            // Render the textured quad.
            renderer.renderQuad();

            // Draw pose overlays on top (scaling from inference to display resolution).
            poseEstimator.drawPosesGL(dispFrame, output);

            // Swap buffers.
            SDL_GL_SwapWindow(renderer.getWindow());

            lastTime = chrono::steady_clock::now();
        }
    }
    catch (const exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }

    return 0;
}
