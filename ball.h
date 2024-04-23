#pragma once

#include <memory>
#include <string>
#include <stdio.h>
#include <box2d/box2d.h>
#include <opencv2/core/mat.hpp>

#define WORLD_SCALE 100.0

class Ball {

public:
    Ball(const b2Vec2& position, b2World& world);
    Ball(const Ball&);
    Ball& operator=(const Ball&);
    ~Ball();
    void update(float delta);
    void impulse(const b2Vec2& force);
    b2Vec2 getPosition() const {return _body->GetPosition();};
    void setPosition(const b2Vec2& pos) const {
        _body->SetLinearVelocity(b2Vec2(0,0));
        _body->SetTransform(pos,_body->GetAngle());
    };
    const b2Body* getBody() {return _body;};
    void render(cv::Mat& image);
private:
    const float _density{10.00f};
    const float _friction{0.0f};
    const float _restitution{0.0f};
    const float _speed{4.0f};
    const float _radius{3.0f/WORLD_SCALE};
    float _timeAlive {0};
    b2Body* _body;
};
