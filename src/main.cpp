#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace saliency;

int main() {
    //Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("original", src);
    GaussianBlur(src,src,Size(5,5),2.5);
    Mat sal,fn,markers,bg;

    Ptr<StaticSaliencySpectralResidual> spectral = StaticSaliencySpectralResidual::create();
    Ptr<StaticSaliencyFineGrained> fine = StaticSaliencyFineGrained::create();
    spectral->computeSaliency(src, sal);
    sal.convertTo(sal, CV_8UC1, 255);
    fine->computeSaliency(src, fn);
    fn.convertTo(fn, CV_8UC1, 255);
    imshow("spectral",sal);
    imshow("fine",fn);
    threshold(fn,fn,0,255,THRESH_BINARY|THRESH_OTSU);
    imshow("otsu_fine",fn);
    erode(fn,fn, getStructuringElement(MORPH_RECT,Size(3,3)));
    dilate(fn,fn, getStructuringElement(MORPH_RECT,Size(3,3)));
    dilate(fn,fn, getStructuringElement(MORPH_RECT,Size(3,3)));
    dilate(fn,fn, getStructuringElement(MORPH_RECT,Size(3,3)));
    imshow("med_fine",fn);

    //background markers
    bitwise_not(fn,bg);
    imshow("bg",bg);

    //foreground markers
    adaptiveThreshold(sal, markers, 255, BORDER_REPLICATE, THRESH_BINARY, 11, 0);
    erode(markers, markers, getStructuringElement(MORPH_RECT, Size(2, 2)));
    //bitwise and to remove non foreground markers + threshold to have a binary inage
    bitwise_and(markers, fn, markers);
    imshow("saliency", markers);

    //watershed
    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    // Visualize the final image
    imshow("Final Result", dst);

    waitKey(0);
    return 0;
}