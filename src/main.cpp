#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    //Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("origin", src);

    Mat src_gray;
    Mat dst, detected_edges;
    int lowThreshold = 70;
    const int max_lowThreshold = 200;
    const int kernel_size = 3;
    const char* window_name = "Edge Map";

    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    GaussianBlur( src_gray, detected_edges, Size(3,3),2);
    Canny( detected_edges, detected_edges, lowThreshold, max_lowThreshold, kernel_size );
    dst = Scalar::all(0);
    src_gray.copyTo( dst, detected_edges);
    //dilate(dst,dst, getStructuringElement(MORPH_RECT,Size(2,2)));

    Mat labels,comp,centroids;
    int nLabels = connectedComponentsWithStats(dst,labels,comp,centroids);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }

    src.convertTo(src,CV_8UC3);
    for(int r = 0; r < src.rows; ++r){
        for(int c = 0; c < src.cols; ++c){
            int label = labels.at<int>(r, c);
            Vec3b &pixel = src.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }
    imshow( window_name, src );
    waitKey(0);
    return 0;
}