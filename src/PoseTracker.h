#ifndef POSE_TRACKER_H
#define POSE_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include <cmath>

using namespace cv;
using namespace std;

struct PoseData {
    int id;
    Point2f center;
};

class PoseTracker {
public:
    PoseTracker();
    // Compute centers for valid poses and assign persistent IDs by nearest-neighbor matching.
    vector<int> trackPoses(const float* output, int numPoses, int inpWidth, int inpHeight,
                           float poseThreshold, float keypointThreshold);
private:
    vector<PoseData> prevPoses;
    int nextId;
};

#endif // POSE_TRACKER_H
