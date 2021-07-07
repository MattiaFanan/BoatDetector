#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace ximgproc::segmentation;

int main() {
    //Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    //Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");

    Mat src = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("origin", src);
    GaussianBlur(src,src,Size(3,3),2);

    //contour
    Mat src_gray;
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    Mat edges(src_gray.rows,src_gray.cols,CV_8UC1,Scalar::all(0));
    vector<vector<Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
    Canny(src_gray,src_gray,50,3*50);
    findContours( src_gray, contours, hierarchy,RETR_TREE, CHAIN_APPROX_SIMPLE );
    for( int i = 0; i< contours.size(); i++ ){
        Scalar color = Scalar( 255,255,255);
        drawContours( edges, contours,i, color,0, 8, hierarchy );
    }
    imshow( "largest Contour", edges );



    Mat res; //Image after segmentation
    int spatialRad = 20; //Spatial window size
    int colorRad = 20; //Color window size
    int maxPyrLevel = 1; //Number of pyramid layers
    pyrMeanShiftFiltering( src, res, spatialRad, colorRad, maxPyrLevel); //Color clustering smoothing filter
    imshow("res",res);
    RNG rng = theRNG();
    Mat mask( res.rows+2, res.cols+2, CV_8UC1, Scalar::all(0) ); //mask
    //edges.copyTo(mask(Rect(1,1,edges.cols,edges.rows)));//copy edges inside mask mask has additional pixels
    for( int y = 0; y < res.rows; y++ ){
        for( int x = 0; x < res.cols; x++ ){
            if( mask.at<uchar>(y+1, x+1) == 0){//Non-zero is 1, which means that it has been filled and no longer processed
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( res, mask, Point(x,y), newVal, 0, Scalar::all(5), Scalar::all(5) ); //Perform flood fill
            }
        }
    }
    imshow("meanShift image segmentation", res );

    //connected components



    waitKey(0);
    return 0;
}