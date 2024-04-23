#include "ball.h"

#include <iostream>
#include <math.h>
#include <opencv2/imgproc.hpp>

Ball::Ball(const b2Vec2& position, b2World& world)
{
    // Create a dynamic body
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody;
    bodyDef.position.Set(position.x/WORLD_SCALE, position.y/WORLD_SCALE);
    _body = world.CreateBody(&bodyDef);

    b2FixtureDef fixtureDef;

    b2CircleShape ballShape;
    ballShape.m_radius = _radius;

    fixtureDef.shape = &ballShape;
    fixtureDef.density = _density;
    fixtureDef.friction = _friction;
    fixtureDef.restitution = _restitution;

    _body->CreateFixture(&fixtureDef);
    _body->SetFixedRotation(true);
}

void Ball::impulse(const b2Vec2& force) {
    _body->SetLinearVelocity(force);
}

void Ball::render(cv::Mat& image) {
    b2Vec2 position = _body->GetPosition();
    int pixelX = static_cast<int>(position.x * WORLD_SCALE);
    int pixelY = static_cast<int>(position.y * WORLD_SCALE);
    int pixelRadius = 50;
    cv::circle(image, cv::Point(pixelX, pixelY), pixelRadius, cv::Scalar(0, 255, 255), -1);
}

void Ball::update(float delta) {
    _timeAlive += delta;
}

Ball::~Ball() {
}
