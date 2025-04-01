#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <chrono>

using namespace std;
using namespace chrono;
random_device rd;
mt19937 gen(rd());

class Point
{
public:
    double X, Y;
    Point *centroid;
    Point(int ll = 0, int ul = 100)
    {
        uniform_real_distribution<double> dist(ll, ul);
        X = dist(gen);
        Y = dist(gen);
    }
};

double euc(Point p1, Point p2)
{
    return sqrt(pow(p1.X - p2.X, 2) + pow(p1.Y - p2.Y, 2));
}

class K_Means
{
public:
    vector<Point> arr;
    vector<Point> centroids;

    int N, K;
    int LL, UL;
    K_Means(int n, int ll = 0, int ul = 100, int k = 2)
    {
        N = n;
        LL = ll;
        UL = ul;
        K = k;
    }

    void assign_centroids()
    {
        for (int i = 0; i < N; i++)
        {
            double min = euc(arr[i], centroids[0]);
            Point *cent = &centroids[0];
            for (int j = 1; j < K; j++)
            {
                double new_d = euc(arr[i], centroids[j]);
                if (new_d < min)
                {
                    min = new_d;
                    cent = &centroids[j];
                }
            }
            arr[i].centroid = cent;
        }
    }

    void reassign()
    {
        for (auto &centroid : centroids)
        {
            centroid.X = 0;
            centroid.Y = 0;
        }

        vector<int> count(K, 0);

        for (auto &point : arr)
        {
            for (int i = 0; i < K; i++)
            {
                if (&centroids[i] == point.centroid)
                {
                    centroids[i].X += point.X;
                    centroids[i].Y += point.Y;
                    count[i]++;
                }
            }
        }

        for (int i = 0; i < K; i++)
        {
            if (count[i] > 0)
            {
                centroids[i].X /= count[i];
                centroids[i].Y /= count[i];
            }
        }
    }

    void random_cents()
    {
        uniform_int_distribution<int> dist(0, N - 1);
        unordered_set<int> selected_indices;

        while (selected_indices.size() < K)
        {
            int a = dist(gen);
            if (selected_indices.insert(a).second)
            {
                centroids.push_back(arr[a]);
            }
        }
    }

    void generate_points()
    {
        for (int i = 0; i < N; i++)
        {
            Point p(LL, UL);
            arr.emplace_back(p);
        }
    }

    void display()
    {
        FILE *gnuplotPipe = _popen("gnuplot -persist", "w");

        if (gnuplotPipe)
        {
            fprintf(gnuplotPipe, "set title 'K-Means Clustering'\n");
            fprintf(gnuplotPipe, "set xlabel 'X-axis'\n");
            fprintf(gnuplotPipe, "set ylabel 'Y-axis'\n");

            const char *colors[] = {"red", "green", "blue", "orange", "purple", "cyan", "brown", "pink", "gray", "magenta", "violet"};

            fprintf(gnuplotPipe, "plot ");

            for (int i = 0; i < K; i++)
            {
                if (i > 0)
                    fprintf(gnuplotPipe, ", ");
                fprintf(gnuplotPipe, "'-' using 1:2 with points pointtype 7 pointsize 0.5 lc rgb '%s' title 'Cluster %d'", colors[i % 10], i);
            }

            fprintf(gnuplotPipe, ", '-' using 1:2 with points pointtype 7 pointsize 1.5 lc rgb 'black' title 'Centroids'\n");

            for (int i = 0; i < K; i++)
            {
                for (auto &p : arr)
                {
                    if (p.centroid == &centroids[i])
                    {
                        fprintf(gnuplotPipe, "%f %f\n", p.X, p.Y);
                    }
                }
                fprintf(gnuplotPipe, "e\n");
            }

            for (auto &c : centroids)
            {
                fprintf(gnuplotPipe, "%f %f\n", c.X, c.Y);
            }
            fprintf(gnuplotPipe, "e\n");

            fflush(gnuplotPipe);
            _pclose(gnuplotPipe);
        }
        else
        {
            cerr << "Could not open Gnuplot!" << endl;
        }
    }
};

int main()
{

    cout << "Enter number of centroids: " << endl;
    int k1;
    cin >> k1;
    int total;
    cout << "Enter total number of datapoints: " << endl;
    cin >> total;
    auto start = high_resolution_clock::now(); // Start time

    K_Means k(total, 0, 100, k1);
    k.generate_points();
    k.random_cents();
    k.assign_centroids();

    for (int i = 0; i < 5; i++)
    {
        k.reassign();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Execution Time: " << duration.count() << " ms" << endl;

    k.display();

    return 0;
}