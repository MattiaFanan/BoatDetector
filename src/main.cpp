#include <iostream>
#include <opencv4/opencv2/saliency.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    //cvtColor(img, img, COLOR_RGB2GRAY);
    /*
    Ptr<saliency::StaticSaliencyFineGrained> pointer = saliency::StaticSaliencyFineGrained::create();
    Mat out;
    Mat threshold_out;
    pointer->computeSaliency(img, out);
    out.convertTo(out, CV_8UC1, 255);
    threshold(out, threshold_out, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", threshold_out);
    waitKey(0);
     */
    Ptr<saliency::ObjectnessBING> pointer = saliency::ObjectnessBING::create();
    pointer->setTrainingPath()
    vector<Vec4i> ROIs;
    Mat out;
    Scalar color = Scalar(0,0, 255);
    img.copyTo(out);
    pointer->computeSaliency(img, ROIs);

    for(Vec4i ROI : ROIs){
        rectangle(out, Point(ROI[0], ROI[1]), Point(ROI[2], ROI[3]), color);
    }

    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", out);
    waitKey(0);
    return 0;
}
