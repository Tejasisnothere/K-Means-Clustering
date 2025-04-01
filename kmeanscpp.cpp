
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;
struct Point {
    float x, y;
    int cluster;
};
float distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
void kMeansClustering(vector<Point> &points, int k) {
    vector<Point> centroids(k);
    for (int i = 0; i < k; i++)
        centroids[i] = points[i];
    bool changed;
    do {
        changed = false;
        for (auto &p : points) {
            float minDist = numeric_limits<float>::max();
            int clusterIndex = -1;
            for (int i = 0; i < k; i++) {
                float dist = distance(p, centroids[i]);
                if (dist < minDist) {
                    minDist = dist;
                    clusterIndex = i;
                }
            }
            if (p.cluster != clusterIndex) {
                p.cluster = clusterIndex;
                changed = true;
            }
        }
        vector<Point> newCentroids(k);
        vector<int> count(k, 0);
        for (auto &p : points) {
            newCentroids[p.cluster].x += p.x;
            newCentroids[p.cluster].y += p.y;
            count[p.cluster]++;
        }
        for (int i = 0; i < k; i++) {
            if (count[i] != 0) {
                newCentroids[i].x /= count[i];
                newCentroids[i].y /= count[i];
            }
        }
        centroids = newCentroids;
    } while (changed);
}
int main() {
    int k = 2;
    vector<Point> points = { {1.0, 1.0, -1}, {1.5, 2.0, -1}, {3.0, 4.0, -1},
                             {5.0, 7.0, -1}, {3.5, 5.0, -1}, {4.5, 5.0, -1}, {3.5, 4.5, -1} };

    kMeansClustering(points, k);
    for (auto &p : points) {
        cout << "Point (" << p.x << ", " << p.y << ") in Cluster " << p.cluster + 1 << endl;
    }
    return 0;
}
