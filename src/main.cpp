#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <SDL.h>
#include <SDL_opengl.h>
#include "VideoInputWrapper.h"
#include "PoseEstimator.h"
#include "OpenGLRenderer.h"
#include "SuitOverlay.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    try {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

        if (SDL_Init(SDL_INIT_VIDEO) < 0)
            throw runtime_error(string("SDL_Init Error: ") + SDL_GetError());

        // Default video source is "0" (webcam); can be overridden via command-line.
        string source = "0";
        if (argc > 1)
            source = argv[1];
        VideoInputWrapper videoInput(source);

        int inputW = static_cast<int>(videoInput.getFrameWidth());
        int inputH = static_cast<int>(videoInput.getFrameHeight());
        cout << "Input resolution: " << inputW << "x" << inputH << endl;

        SDL_DisplayMode dm;
        if (SDL_GetCurrentDisplayMode(0, &dm) != 0)
            throw runtime_error(string("SDL_GetCurrentDisplayMode Error: ") + SDL_GetError());
        int desktopW = dm.w;
        int desktopH = dm.h;
        cout << "Desktop resolution: " << desktopW << "x" << desktopH << endl;

        float scaleFactor = min(1.0f, min(static_cast<float>(desktopW) / inputW, static_cast<float>(desktopH) / inputH));
        int windowW = static_cast<int>(inputW * scaleFactor);
        int windowH = static_cast<int>(inputH * scaleFactor);
        cout << "Window resolution: " << windowW << "x" << windowH << endl;

        OpenGLRenderer renderer(windowW, windowH);

        // Load suit textures.
        SuitImages suit;
        suit.face = loadSuitTexture("../assets/face.png");
        suit.torso = loadSuitTexture("../assets/torso.png");
        suit.leftArm = loadSuitTexture("../assets/left_arm.png");
        suit.rightArm = loadSuitTexture("../assets/right_arm.png");
        suit.leftForearm = loadSuitTexture("../assets/left_forearm.png");
        suit.rightForearm = loadSuitTexture("../assets/right_forearm.png");
        suit.leftLeg = loadSuitTexture("../assets/left_leg.png");
        suit.rightLeg = loadSuitTexture("../assets/right_leg.png");
        suit.leftLowerLeg = loadSuitTexture("../assets/left_lower_leg.png");
        suit.rightLowerLeg = loadSuitTexture("../assets/right_lower_leg.png");

        // Define scale factors.
        SuitScaleFactors scales;
        scales.face = 3.5f;
        scales.torso = 1.0f;
        scales.leftArm = 1.0f;
        scales.rightArm = 1.0f;
        scales.leftForearm = 1.0f;
        scales.rightForearm = 1.0f;
        scales.leftLeg = 1.0f;
        scales.rightLeg = 1.0f;
        scales.leftLowerLeg = 1.0f;
        scales.rightLowerLeg = 1.0f;

        string modelPath = "../assets/lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";
        PoseEstimator poseEstimator(modelPath, true, 192, 192);

        bool drawKeypoints = false;
        bool running = true;
        SDL_Event event;
        Mat fullFrame, inferenceFrame, dispFrame, rgbDispFrame;

        while (running) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    running = false;
                else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED)
                    renderer.updateViewport();
                else if (event.type == SDL_KEYDOWN) {
                    if (event.key.keysym.sym == SDLK_ESCAPE)
                        running = false;
                    else if (event.key.keysym.sym == SDLK_l)
                        drawKeypoints = !drawKeypoints;
                }
            }
            if (!videoInput.getFrame(fullFrame))
                continue;
            // Mirror frame horizontally.
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
            poseEstimator.drawPosesGL(dispFrame, output, suit, scales, drawKeypoints);
            SDL_GL_SwapWindow(renderer.getWindow());
        }
    }
    catch (const exception &e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }
    return 0;
}
