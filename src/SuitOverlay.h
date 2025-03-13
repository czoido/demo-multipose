#ifndef SUIT_OVERLAY_H
#define SUIT_OVERLAY_H

#include <opencv2/opencv.hpp>
#include <SDL_opengl.h>
#include <stdexcept>
#include <string>

using namespace cv;
using namespace std;

// SuitTexture holds an OpenGL texture ID and its original dimensions.
struct SuitTexture {
    GLuint textureID;
    int width;
    int height;
};

// SuitScaleFactors holds scale multipliers for each suit part.
struct SuitScaleFactors {
    float face;
    float torso;
    float leftArm;
    float rightArm;
    float leftForearm;
    float rightForearm;
    float leftLeg;
    float rightLeg;
    float leftLowerLeg;
    float rightLowerLeg;
};

// SuitImages holds textures for various suit parts.
struct SuitImages {
    SuitTexture face;
    SuitTexture torso;
    SuitTexture leftArm;
    SuitTexture rightArm;
    SuitTexture leftForearm;
    SuitTexture rightForearm;
    SuitTexture leftLeg;
    SuitTexture rightLeg;
    SuitTexture leftLowerLeg;
    SuitTexture rightLowerLeg;
};

// Loads an image file into an OpenGL texture.
SuitTexture loadSuitTexture(const string& filename);

// Draws a textured quad without rotation.
void drawTextureRect(float centerX, float centerY, float width, float height, const SuitTexture &tex);

// Draws a textured quad rotated by the given angle (in degrees).
void drawRotatedTextureRect(float centerX, float centerY, float width, float height, float angle, const SuitTexture &tex);

#endif // SUIT_OVERLAY_H
