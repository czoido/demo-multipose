#include "SuitOverlay.h"

SuitTexture loadSuitTexture(const string& filename) {
    Mat img = imread(filename, IMREAD_UNCHANGED);
    if (img.empty())
        throw runtime_error("Failed to load suit image: " + filename);
    if (img.channels() == 4)
        cvtColor(img, img, COLOR_BGRA2RGBA);
    SuitTexture tex;
    glGenTextures(1, &tex.textureID);
    glBindTexture(GL_TEXTURE_2D, tex.textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
    tex.width = img.cols;
    tex.height = img.rows;
    return tex;
}

void drawTextureRect(float centerX, float centerY, float width, float height, const SuitTexture &tex) {
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

void drawRotatedTextureRect(float centerX, float centerY, float width, float height, float angle, const SuitTexture &tex) {
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
