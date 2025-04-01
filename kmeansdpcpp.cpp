#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <chrono>

class Point {
public:
    double X, Y;
    int cluster_id;

    Point(double x = 0, double y = 0) : X(x), Y(y), cluster_id(-1) {}
};

// Euclidean Distance Function
double euc(const Point& p1, const Point& p2) {
    return sycl::sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y));
}

class KMeans {
public:
    std::vector<Point> points;
    std::vector<Point> centroids;
    int N, K;

    KMeans(int n, int k) : N(n), K(k) {}

    void generate_points() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 100.0);
        
        for (int i = 0; i < N; i++) {
            points.emplace_back(dist(gen), dist(gen));
        }
    }

    void initialize_centroids() {
        std::unordered_set<int> selected_indices;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, N - 1);
        
        while (selected_indices.size() < K) {
            int idx = dist(gen);
            if (selected_indices.insert(idx).second) {
                centroids.push_back(points[idx]);
            }
        }
    }

    void assign_clusters(sycl::queue& q) {
        sycl::buffer<Point, 1> points_buf(points.data(), sycl::range<1>(N));
        sycl::buffer<Point, 1> centroids_buf(centroids.data(), sycl::range<1>(K));
        
        q.submit([&](sycl::handler& h) {
            auto points_acc = points_buf.get_access<sycl::access::mode::read_write>(h);
            auto centroids_acc = centroids_buf.get_access<sycl::access::mode::read>(h);
            int local_K = K;  // Local copy

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                double min_dist = euc(points_acc[i], centroids_acc[0]);
                int closest = 0;
                
                for (int j = 1; j < local_K; j++) {
                    double dist = euc(points_acc[i], centroids_acc[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest = j;
                    }
                }
                points_acc[i].cluster_id = closest;
            });
        }).wait();
    }

    void update_centroids(sycl::queue& q) {
        std::vector<Point> new_centroids(K, Point(0, 0));
        std::vector<int> count(K, 0);
        
        sycl::buffer<Point, 1> points_buf(points.data(), sycl::range<1>(N));
        sycl::buffer<Point, 1> centroids_buf(new_centroids.data(), sycl::range<1>(K));
        sycl::buffer<int, 1> count_buf(count.data(), sycl::range<1>(K));

        q.submit([&](sycl::handler& h) {
            auto points_acc = points_buf.get_access<sycl::access::mode::read>(h);
            auto centroids_acc = centroids_buf.get_access<sycl::access::mode::read_write>(h);
            auto count_acc = count_buf.get_access<sycl::access::mode::read_write>(h);
            
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                int cid = points_acc[i].cluster_id;
                sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device> 
                    atomic_x(centroids_acc[cid].X), atomic_y(centroids_acc[cid].Y);
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> 
                    atomic_count(count_acc[cid]);
                
                atomic_x.fetch_add(points_acc[i].X);
                atomic_y.fetch_add(points_acc[i].Y);
                atomic_count.fetch_add(1);
            });
        }).wait();

        // Compute final centroid positions on the host
        for (int i = 0; i < K; i++) {
            if (count[i] > 0) {
                new_centroids[i].X /= count[i];
                new_centroids[i].Y /= count[i];
            }
        }

        centroids = new_centroids;
    }

    void run(sycl::queue& q, int iterations) {
        for (int i = 0; i < iterations; i++) {
            assign_clusters(q);
            update_centroids(q);
        }
    }

    
};

int main() {
    sycl::queue q(sycl::default_selector_v);

    
    
    auto start = std::chrono::high_resolution_clock::now();
    
    KMeans kmeans(10000, 50);
    kmeans.generate_points();
    kmeans.initialize_centroids();
    kmeans.run(q, 5);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;
    
    
    return 0;
}