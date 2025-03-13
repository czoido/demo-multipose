#include "PoseTracker.h"

PoseTracker::PoseTracker() : nextId(0) {}

vector<int> PoseTracker::trackPoses(const float* output, int numPoses, int inpWidth, int inpHeight,
                                    float poseThreshold, float keypointThreshold) {
    vector<PoseData> currentPoses;
    for (int p = 0; p < numPoses; p++) {
        const float* pose = output + (56 * p);
        float score = pose[55];
        if (score < poseThreshold)
            continue;
        Point2f center(0, 0);
        int count = 0;
        for (int k = 0; k < 17; k++) {
            const float* kp = pose + 3 * k;
            if (kp[2] < keypointThreshold)
                continue;
            center.x += kp[1] * inpWidth;
            center.y += kp[0] * inpHeight;
            count++;
        }
        if (count > 0) {
            center.x /= count;
            center.y /= count;
            currentPoses.push_back({-1, center});
        }
    }
    vector<bool> used(prevPoses.size(), false);
    for (auto &curr : currentPoses) {
        float bestDist = numeric_limits<float>::max();
        int bestIdx = -1;
        for (int i = 0; i < prevPoses.size(); i++) {
            if (used[i])
                continue;
            float dist = norm(curr.center - prevPoses[i].center);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }
        if (bestIdx != -1 && bestDist < 50.0f) {
            curr.id = prevPoses[bestIdx].id;
            used[bestIdx] = true;
        } else {
            curr.id = nextId++;
        }
    }
    prevPoses = currentPoses;
    vector<int> result(numPoses, -1);
    int j = 0;
    for (int p = 0; p < numPoses; p++) {
        const float* pose = output + (56 * p);
        if (pose[55] < poseThreshold)
            continue;
        result[p] = currentPoses[j].id;
        j++;
    }
    return result;
}
