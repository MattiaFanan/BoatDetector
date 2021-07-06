#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace ximgproc::segmentation;


void applyCustomColormap(const Mat1i& src, Mat3b& dst);
int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("origin", img);

    Mat1i out;
    vector<Rect> ROIs;
    Scalar color = Scalar(0,255,0);
    double sigma=0.5;
    float k=900;
    int min_size=100;
    Ptr<GraphSegmentation> segmenter = createGraphSegmentation(sigma,k,min_size);
    segmenter->processImage(img,out);

    Mat3b result;
    applyCustomColormap(out, result);

    imshow("Result", result);
    waitKey();

    return 0;
}

void applyCustomColormap(const Mat1i& src, Mat3b& dst)
{
    // Create JET colormap

    double m;
    minMaxLoc(src, nullptr, &m);
    m++;

    int n = ceil(m / 4);
    Mat1d u(n*3-1, 1, double(1.0));

    for (int i = 1; i <= n; ++i) {
        u(i-1) = double(i) / n;
        u((n*3-1) - i) = double(i) / n;
    }

    vector<double> g(n * 3 - 1, 1);
    vector<double> r(n * 3 - 1, 1);
    vector<double> b(n * 3 - 1, 1);
    for (int i = 0; i < g.size(); ++i)
    {
        g[i] = ceil(double(n) / 2) - (int(m)%4 == 1 ? 1 : 0) + i + 1;
        r[i] = g[i] + n;
        b[i] = g[i] - n;
    }

    g.erase(remove_if(g.begin(), g.end(), [m](double v){ return v > m;}), g.end());
    r.erase(remove_if(r.begin(), r.end(), [m](double v){ return v > m; }), r.end());
    b.erase(remove_if(b.begin(), b.end(), [](double v){ return v < 1.0; }), b.end());

    Mat1d cmap(m, 3, double(0.0));
    for (int i = 0; i < r.size(); ++i) { cmap(int(r[i])-1, 2) = u(i); }
    for (int i = 0; i < g.size(); ++i) { cmap(int(g[i])-1, 1) = u(i); }
    for (int i = 0; i < b.size(); ++i) { cmap(int(b[i])-1, 0) = u(u.rows - b.size() + i); }

    Mat3d cmap3 = cmap.reshape(3);

    Mat3b colormap;
    cmap3.convertTo(colormap, CV_8U, 255.0);


    // Apply color mapping
    dst = Mat3b(src.rows, src.cols, Vec3b(0,0,0));
    for (int r = 0; r < src.rows; ++r)
    {
        for (int c = 0; c < src.cols; ++c)
        {
            dst(r, c) = colormap(src(r,c));
        }
    }
}