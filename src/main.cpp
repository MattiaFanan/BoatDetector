#include <iostream>
#include <opencv2/saliency.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace saliency;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/02.png");
    cvtColor(img, img, COLOR_RGB2GRAY);
    Ptr<saliency::StaticSaliencySpectralResidual> pointer = saliency::StaticSaliencySpectralResidual::create();
    Mat out;
    Mat threshold_out;
    pointer->computeSaliency(img, out);
    out.convertTo(out, CV_8UC1, 255);
    threshold(out, threshold_out, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", threshold_out);
    waitKey(0);

    /*
    int erosion_size = 1;
    Mat element = getStructuringElement( MORPH_RECT,Size( 2*erosion_size + 1, 2*erosion_size+1 ));
    morphologyEx( threshold_out, threshold_out, MORPH_DILATE, element );
    //erode(threshold_out,threshold_out,element);*/


    Mat labelImage, stat, centroid;
    int nLabels = connectedComponentsWithStats(threshold_out, labelImage, stat,centroid,8);

    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    Mat dst(img.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }
    imshow( "Connected Components", dst );
    waitKey(0);
    return 0;
}
