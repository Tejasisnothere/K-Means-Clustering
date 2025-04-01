
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace sycl;
using namespace std;
struct Point {
    float x, y;
    int cluster;
};
float distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
void kMeansClustering(vector<Point> &points, int k) {
    queue q;
    buffer<Point, 1> points_buf(points.data(), range<1>(points.size()));
    buffer<Point, 1> centroids_buf(k);
    q.submit([&](handler &h) {
        auto points_acc = points_buf.get_access<access::mode::read_write>(h);
        auto centroids_acc = centroids_buf.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(k), [=](id<1> i) {
            centroids_acc[i] = points_acc[i];
        });
    });
    bool changed;
    do {
        changed = false;
        buffer<bool, 1> changed_buf(&changed, range<1>(1));
        q.submit([&](handler &h) {
            auto points_acc = points_buf.get_access<access::mode::read_write>(h);
            auto centroids_acc = centroids_buf.get_access<access::mode::read>(h);
            auto changed_acc = changed_buf.get_access<access::mode::write>(h);
            h.parallel_for(range<1>(points.size()), [=](id<1> i) {
                float minDist = numeric_limits<float>::max();
                int clusterIndex = -1;
                for (int j = 0; j < k; j++) {
                    float dist = distance(points_acc[i], centroids_acc[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        clusterIndex = j;
                    }
                }
                if (points_acc[i].cluster != clusterIndex) {
                    points_acc[i].cluster = clusterIndex;
                    *changed_acc = true;
                }
            });
        });
        vector<Point> newCentroids(k);
        vector<int> count(k, 0);
        q.submit([&](handler &h) {
            auto points_acc = points_buf.get_access<access::mode::read>(h);
            h.parallel_for(range<1>(points.size()), [=](id<1> i) {
                atomic<float> xSum(0.0f), ySum(0.0f);
                atomic<int> c(0);
                xSum.fetch_add(points_acc[i].x);
                ySum.fetch_add(points_acc[i].y);
                c.fetch_add(1);

                newCentroids[points_acc[i].cluster].x = xSum.load() / c.load();
                newCentroids[points_acc[i].cluster].y = ySum.load() / c.load();
            });
        });
        q.submit([&](handler &h) {
            auto centroids_acc = centroids_buf.get_access<access::mode::write>(h);
            h.parallel_for(range<1>(k), [=](id<1> i) {
                centroids_acc[i] = newCentroids[i];
            });
        });
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