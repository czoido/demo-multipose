// main_multipose.cpp
// Refactored multipose tracking example using OpenCV, TensorFlow Lite, SDL2 and OpenGL.
// The program captures webcam frames, runs multipose estimation and draws keypoints and connections.
// All drawing is done with OpenGL.

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

using namespace std;
using namespace cv;

// Global variables for pose drawing (used by PoseEstimator)
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

//------------------------------------------------------------
// Class that wraps OpenCV VideoCapture
//------------------------------------------------------------
class WebcamCapture {
public:
    WebcamCapture(int device = 0) : cap(device) {
        if (!cap.isOpened())
            throw runtime_error("Failed to open webcam");
    }
    bool getFrame(Mat &frame) {
        cap >> frame;
        return !frame.empty();
    }
private:
    VideoCapture cap;
};

//------------------------------------------------------------
// Class that handles TFLite multipose estimation and pose drawing
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

    // Runs inference on the provided image.
    // Assumes that 'input' is in BGR format and will be resized if necessary.
    float* runInference(const Mat &input) {
        Mat resized;
        if (input.cols != inputWidth || input.rows != inputHeight)
            resize(input, resized, Size(inputWidth, inputHeight));
        else
            resized = input;
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), resized.data,
               resized.total() * resized.elemSize());
        if (interpreter->Invoke() != kTfLiteOk)
            cerr << "Inference failed" << endl;
        return interpreter->typed_output_tensor<float>(0);
    }

    // Draws pose keypoints and connections using OpenGL calls.
    // 'image' is used only for its width and height.
    void drawPosesGL(const Mat &image, float* output) {
        int width = image.cols;
        int height = image.rows;
        // Assume output tensor shape is [1, numPoses, 56]
        int numPoses = interpreter->tensor(interpreter->outputs()[0])->dims->data[1];
        // Draw keypoints
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
                float x = keypoint[1] * width;
                float y = keypoint[0] * height;
                glVertex2f(x, y);
            }
        }
        glEnd();
        // Draw connections
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
                float x1 = kp1[1] * width;
                float y1 = kp1[0] * height;
                float x2 = kp2[1] * width;
                float y2 = kp2[0] * height;
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
// Class that wraps SDL2 and OpenGL for rendering
//------------------------------------------------------------
class OpenGLRenderer {
public:
    OpenGLRenderer(int width, int height)
            : windowWidth(width), windowHeight(height)
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
        // Create texture; initial texture size is set to window size
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

    // Update the viewport and projection (call on window resize)
    void updateViewport() {
        SDL_GetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    // Update the OpenGL texture with the given frame (assumes frame is in RGB)
    void updateTexture(const Mat &frame) {
        glBindTexture(GL_TEXTURE_2D, textureID);
        // If frame dimensions differ from current texture, reallocate storage
        if (frame.cols != windowWidth || frame.rows != windowHeight) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
        }
    }

    // Render the textured quad covering the entire window.
    void renderQuad() {
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
};

//------------------------------------------------------------
// Main function: integrate webcam, multipose estimation and rendering
//------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        // Create webcam capture instance
        WebcamCapture webcam(0);

        // Create multipose estimator (model path, multiPose flag, input dimensions)
        string modelPath = "../lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";
        PoseEstimator poseEstimator(modelPath, true, 192, 192);

        // Create OpenGL renderer; initial window size set to model input size
        OpenGLRenderer renderer(192, 192);

        bool running = true;
        SDL_Event event;
        Mat frame, rgbFrame, resizedFrame;
        auto lastTime = chrono::steady_clock::now();

        while (running) {
            // Process SDL events
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    running = false;
                else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    renderer.updateViewport();
                }
            }

            // Get frame from webcam
            if (!webcam.getFrame(frame))
                continue;
            // Mirror the frame horizontally
            flip(frame, frame, 1);
            // Resize frame to model input size
            resize(frame, resizedFrame, Size(192, 192));

            // Run multipose inference
            float* output = poseEstimator.runInference(resizedFrame);

            // Convert resized frame from BGR to RGB for texture update
            cvtColor(resizedFrame, rgbFrame, COLOR_BGR2RGB);
            if (!rgbFrame.isContinuous())
                rgbFrame = rgbFrame.clone();

            // Update texture in renderer with the current frame
            renderer.updateTexture(rgbFrame);

            // Clear screen and render the textured quad
            glClear(GL_COLOR_BUFFER_BIT);
            renderer.renderQuad();

            // Draw pose overlays (keypoints and connections) over the image
            poseEstimator.drawPosesGL(rgbFrame, output);

            // Swap buffers once per frame
            SDL_GL_SwapWindow(renderer.getWindow());

            auto now = chrono::steady_clock::now();
            lastTime = now;
        }
    }
    catch (const exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }

    return 0;
}
