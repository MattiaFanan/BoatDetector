#include <iostream>
#include <opencv2/hfs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace hfs;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    imshow("orig",img);

    Ptr<HfsSegment> segm = HfsSegment::create(img.rows, img.cols);

    //segmentation
    Mat img_seg = segm->performSegmentCpu(img);
    imshow("img_seg", img_seg);

    //edges
    int thresh=0;
    Mat edges;
    Canny(img_seg, edges, thresh, 3 * thresh);
    imshow("edges",edges);

    //connected components
    Mat conn;
    bitwise_not(edges,conn);
    erode(conn,conn, getStructuringElement(MORPH_RECT,Size(3,3)));

    Mat labelImage, stat, centroid;
    int nLabels = connectedComponentsWithStats(conn, labelImage, stat, centroid, 8);
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

    /*
    int max_det = 200;
    for(ulong i = 0; i < max_det; i++){
        Vec4i ROI = ROIs[i];
        img.copyTo(img_seg);
        putText(img_seg, to_string(score[i]), Point(ROI[0]+20, ROI[1]+20), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color, 2, LINE_AA);
        rectangle(img_seg, Point(ROI[0], ROI[1]), Point(ROI[2], ROI[3]), color);
        imshow("TMP", img_seg);
        waitKey(0);
    }

    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", img_seg);
    waitKey(0);
     */

    waitKey(0);
    return 0;
}
