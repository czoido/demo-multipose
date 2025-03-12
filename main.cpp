// main_sdl_capture.cpp
// Minimal example: capture webcam video with OpenCV and render it in an SDL2 window using OpenGL.
// This version flips the texture horizontally by reversing the U texture coordinates.
// All comments are in English.

#include <opencv2/opencv.hpp>
#include <SDL.h>
#include <SDL_opengl.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    // Open the default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the webcam." << endl;
        return -1;
    }

    // Capture an initial frame to obtain resolution
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: Captured empty frame." << endl;
        return -1;
    }
    int camWidth = frame.cols;
    int camHeight = frame.rows;
    cout << "Camera resolution: " << camWidth << " x " << camHeight << endl;

    // Use the camera's resolution for our texture and initial window size.
    int windowWidth = camWidth;
    int windowHeight = camHeight;

    // Initialize SDL2 video subsystem
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return -1;
    }

    // Create an SDL window (resizable) with an OpenGL context
    SDL_Window* window = SDL_CreateWindow("Webcam Feed", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          windowWidth, windowHeight,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        cerr << "SDL_CreateWindow Error: " << SDL_GetError() << endl;
        SDL_Quit();
        return -1;
    }

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        cerr << "SDL_GL_CreateContext Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Set up viewport and orthographic projection (0,0 at top-left)
    SDL_GetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Enable 2D texturing and set clear color to black
    glEnable(GL_TEXTURE_2D);
    glClearColor(0, 0, 0, 1);

    // Create an OpenGL texture to hold the webcam image (RGB format)
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Set pixel storage mode (in case rows are not 4-byte aligned)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // Allocate texture storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, camWidth, camHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    bool running = true;
    SDL_Event event;
    auto lastTime = chrono::steady_clock::now();

    while (running) {
        // Handle SDL events, including window resize
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
                windowWidth = event.window.data1;
                windowHeight = event.window.data2;
                glViewport(0, 0, windowWidth, windowHeight);
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity();
            }
        }

        // Capture a new frame from the webcam
        cap >> frame;
        if (frame.empty()) {
            cerr << "Warning: Captured empty frame." << endl;
            continue;
        }

        // Convert from BGR (OpenCV default) to RGB (for OpenGL)
        Mat rgbFrame;
        cvtColor(frame, rgbFrame, COLOR_BGR2RGB);
        if (!rgbFrame.isContinuous())
            rgbFrame = rgbFrame.clone();

        // Update the texture with the current frame data
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camWidth, camHeight, GL_RGB, GL_UNSIGNED_BYTE, rgbFrame.data);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw a textured quad covering the entire window, but flip horizontally by swapping U coordinates
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(windowWidth, 0);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(0, 0);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(0, windowHeight);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(windowWidth, windowHeight);
        glEnd();

        SDL_GL_SwapWindow(window);
        lastTime = chrono::steady_clock::now();
    }

    // Cleanup resources
    cap.release();
    glDeleteTextures(1, &textureID);
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
