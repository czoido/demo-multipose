#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int initialWidth, int initialHeight)
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

OpenGLRenderer::~OpenGLRenderer() {
    glDeleteTextures(1, &textureID);
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void OpenGLRenderer::updateViewport() {
    SDL_GetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void OpenGLRenderer::updateTexture(const Mat &frame) {
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

void OpenGLRenderer::renderQuad() {
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(windowWidth, 0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(windowWidth, windowHeight);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0, windowHeight);
    glEnd();
}

SDL_Window* OpenGLRenderer::getWindow() {
    return window;
}

int OpenGLRenderer::getWidth() const {
    return windowWidth;
}

int OpenGLRenderer::getHeight() const {
    return windowHeight;
}
