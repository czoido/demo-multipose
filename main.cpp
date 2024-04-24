// With code taken from: https://github.com/kleincode/Webcam-Pose-Estimation

#include "ball.h"

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <map>
#include <cstdlib>  // Para rand() y srand()
#include <ctime>    // Para time()

using namespace std;

std::map<int, cv::Point2f> previousHeadPositions;

const vector<cv::Scalar> colors = {
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255)
};

cv::Mat resizeFrame(cv::Mat &frame, cv::Mat &resized, int targetWidth, int targetHeight) {
    int height = frame.size().height;
    int width = frame.size().width;
    double xScale = targetWidth / (double) width;
    double yScale = targetHeight / (double) height;

    cv::Mat cropped;

    if (yScale > xScale) {
        int sourceWidth = cvCeil(targetWidth / yScale);
        int sourcePad = (width - sourceWidth) / 2;

        cropped = frame(cv::Rect(sourcePad, 0, sourceWidth, height));
        cv::resize(cropped, resized, cv::Size(targetWidth, targetHeight));
    } else {
        int sourceHeight = cvCeil(targetHeight / xScale);
        int sourcePad = (height - sourceHeight) / 2;

        cropped = frame(cv::Rect(0, sourcePad, width, sourceHeight));
        cv::resize(cropped, resized, cv::Size(targetWidth, targetHeight));
    }
    return cropped;
}

float *runInterpreter(unique_ptr<tflite::Interpreter> &interpreter, cv::Mat &input) {
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), input.data, input.total() * input.elemSize());
    chrono::steady_clock::time_point start, end;
    start = chrono::steady_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
        cerr << "Inference failed" << endl;
    }
    end = chrono::steady_clock::now();
    float *results = interpreter->typed_output_tensor<float>(0);
    return results;
}

const float poseThreshold = 0.2f, keypointThreshold = 0.2f;
const auto red = cv::Scalar(50, 50, 255);
const std::vector<std::pair<int, int>> connections = {
        {0,  1},
        {0,  2},
        {1,  3},
        {2,  4},
        {5,  6},
        {5,  7},
        {7,  9},
        {6,  8},
        {8,  10},
        {5,  11},
        {6,  12},
        {11, 12},
        {11, 13},
        {13, 15},
        {12, 14},
        {14, 16}};

void drawKeypoints(cv::Mat &target, float *output, int poses, double frameMs) {
    int width = target.size().width;
    int height = target.size().height;

    for (int p = 0; p < poses; p++) {
        float *pose = output + (56 * p);
        float score = pose[55];

        if (score < poseThreshold) continue;

        cv::Scalar poseColor = colors[p % colors.size()];

        if (poses > 1) {
            float ymin = pose[51], xmin = pose[52], ymax = pose[53], xmax = pose[54];
            cv::Rect bbox = cv::Rect(
                    static_cast<int>(width * xmin),
                    static_cast<int>(height * ymin),
                    static_cast<int>(width * (xmax - xmin)),
                    static_cast<int>(height * (ymax - ymin))
            );

            cv::rectangle(target, bbox, poseColor, 2);
            cv::putText(target, to_string(cvRound(100 * score)) + "%",
                        cv::Point(static_cast<int>(width * xmin), static_cast<int>(height * ymin - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, poseColor);
        }

        for (int k = 0; k < 17; k++) {
            float *keypoint = pose + 3 * k;
            float y = keypoint[0], x = keypoint[1], conf = keypoint[2];
            if (conf < keypointThreshold) continue;

            cv::circle(target, cv::Point(static_cast<int>(x * width), static_cast<int>(y * height)), 4, poseColor,
                       cv::FILLED);
        }

        for (const auto &connection: connections) {
            float *keypoint1 = pose + 3 * connection.first;
            float y1 = keypoint1[0], x1 = keypoint1[1], conf1 = keypoint1[2];

            float *keypoint2 = pose + 3 * connection.second;
            float y2 = keypoint2[0], x2 = keypoint2[1], conf2 = keypoint2[2];

            if (conf1 < keypointThreshold || conf2 < keypointThreshold) continue;

            auto p1 = cv::Point(static_cast<int>(x1 * width), static_cast<int>(y1 * height));
            auto p2 = cv::Point(static_cast<int>(x2 * width), static_cast<int>(y2 * height));

            cv::line(target, p1, p2, poseColor, 1);
        }
    }
    cv::putText(target, to_string(cvRound(1000.0 / frameMs)) + " FPS", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255));
}

const float headRadius = 1.0f;
const float impulseStrength = 2.0f;

void updateBallImpulse(cv::Mat &target, float *output, b2World &world, Ball &ball, double deltaTime) {
    int poses = 6; // Asumiendo que hay hasta 6 personas detectadas
    int width = target.size().width;
    int height = target.size().height;

    // Índices de keypoints para cabeza y muñecas (ajustar según tu modelo específico)
    const int headIndex = 0;
    const int leftWristIndex = 9;
    const int rightWristIndex = 10;

    for (int p = 0; p < poses; p++) {
        float *pose = output + (56 * p);  // Asumiendo que cada pose tiene 56 valores (17 keypoints * 3 valores cada uno + 5 adicionales)
        float score = pose[55];  // Score de detección de la pose
        if (score < poseThreshold) continue;

        // Procesar la cabeza y ambas muñecas
        std::vector<int> indices = {headIndex, leftWristIndex, rightWristIndex};
        for (int index : indices) {
            float y = pose[3 * index] * height;
            float x = pose[3 * index + 1] * width;
            cv::Point2f currentPosition(x, y);

            int key = p * 100 + index;  // clave única para cada persona y cada parte del cuerpo
            if (previousHeadPositions.find(key) != previousHeadPositions.end()) {
                cv::Point2f prevPosition = previousHeadPositions[key];
                cv::Point2f velocity = (currentPosition - prevPosition) / static_cast<float>(deltaTime);

                b2Vec2 ballPosition = ball.getPosition();
                ballPosition.x *= WORLD_SCALE;
                ballPosition.y *= WORLD_SCALE;

                float distance = cv::norm(currentPosition - cv::Point2f(ballPosition.x, ballPosition.y));
                if (distance < headRadius * WORLD_SCALE) {
                    b2Vec2 impulse(velocity.x / WORLD_SCALE, velocity.y / WORLD_SCALE);
                    b2Vec2 impulseForce(impulse.x * impulseStrength, -6.0);
                    ball.impulse(impulseForce);
                }
            }
            previousHeadPositions[key] = currentPosition;
        }
    }
}

int main() {
    bool multiPose;
    string modelFile;
    multiPose = true;
    modelFile = "../lite-model_movenet_multipose_lightning_tflite_float16_4.tflite";

    /*
        TFLITE SETUP
    */
    cout << "Loading interpreter" << endl;
    auto model = tflite::FlatBufferModel::BuildFromFile(modelFile.c_str());

    if (!model) {
        throw runtime_error("Failed to load model from " + modelFile);
    }
    cout << "Model loaded: " << modelFile << endl;

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw runtime_error("Failed to allocate tensors");
    }

    auto input = interpreter->inputs()[0];
    auto output = interpreter->outputs()[0];

    auto output_dims_size = interpreter->tensor(output)->dims->size;
    auto output_dims = interpreter->tensor(output)->dims->data;

    auto input_dims_size = interpreter->tensor(input)->dims->size;
    auto input_dims = interpreter->tensor(input)->dims->data;

    cout << "The input tensor has the following dimensions: [" << input_dims;
    for (int d = 1; d < input_dims_size; d++) {
        cout << ", " << input_dims[d];
    }
    cout << "]" << endl;

    if (input_dims_size < 3) {
        throw runtime_error("Input dims should be at least 3d.");
    }

    int inputWidth = 192;
    int inputHeight = 192;

    if (multiPose) {
        interpreter->ResizeInputTensor(0, {1, inputHeight, inputWidth, 3});
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            throw runtime_error("Failed to reallocate tensors");
        }
    }

    cout << "The output tensor has the following dimensions: [" << output_dims;
    for (int d = 1; d < output_dims_size; d++) {
        cout << ", " << output_dims[d];
    }
    cout << "]" << endl;

    int poses = output_dims_size > 1 ? output_dims[1] : 1;
    cout << "Number of detected poses: " << poses << endl;

    /*
        BOX2D
    */

    b2Vec2 gravity(0.0f, 12.0f);
    b2World world(gravity);
    Ball ball(b2Vec2(320.0f, 0.0f), world);

    /*
        OPENCV SETUP
    */

    cv::Mat frame, resized;
    cv::namedWindow("Output");
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "No video stream detected" << endl;
        system("pause");
        return -1;
    }

    auto _time = chrono::steady_clock::now();
    double avgFrameMs = 1000.0 / 30.0;
    srand(time(NULL));

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::flip(frame, frame, 1);

        cv::Mat cropped = resizeFrame(frame, resized, inputWidth, inputHeight);

        float *results = runInterpreter(interpreter, resized);
        auto now = chrono::steady_clock::now();
        auto frameMs = chrono::duration_cast<chrono::milliseconds>(now - _time).count();
        _time = now;
        avgFrameMs += 0.05 * (frameMs - avgFrameMs);
        drawKeypoints(cropped, results, poses, avgFrameMs);
        ball.render(cropped);

        if (ball.getPosition().y > (float) cropped.cols / WORLD_SCALE ||
            ball.getPosition().x < 0.0 || ball.getPosition().x > (float)cropped.rows / WORLD_SCALE) {
            // Calcular un valor aleatorio entre el 20% y el 80% del ancho de la imagen
            float randomX = 0.2f * cropped.cols + static_cast<float>(rand()) / RAND_MAX * (0.6f * cropped.cols);

            // Convertir a unidades de Box2D (WORLD_SCALE)
            b2Vec2 newPosition(randomX / WORLD_SCALE, 0.0f / WORLD_SCALE);
            ball.setPosition(newPosition);
        }
        updateBallImpulse(cropped, results, world, ball, frameMs / 1000.0);
        world.Step(frameMs / 1000.0, 8, 3);
        cv::imshow("Output", cropped);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
