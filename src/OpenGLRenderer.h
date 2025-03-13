#ifndef OPENGL_RENDERER_H
#define OPENGL_RENDERER_H

#include <SDL.h>
#include <SDL_opengl.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

class OpenGLRenderer {
public:
    OpenGLRenderer(int initialWidth, int initialHeight);
    ~OpenGLRenderer();
    void updateViewport();
    void updateTexture(const Mat &frame);
    void renderQuad();
    SDL_Window* getWindow();
    int getWidth() const;
    int getHeight() const;
private:
    SDL_Window* window;
    SDL_GLContext glContext;
    GLuint textureID;
    int windowWidth;
    int windowHeight;
    int textureWidth;
    int textureHeight;
};

#endif // OPENGL_RENDERER_H
