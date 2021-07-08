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

    // ROIs
    int min_area=200;
    vector<Rect> ROIs;
    for(int i=0; i < nLabels; i++){
        if(stat.at<int>(i,CC_STAT_AREA) >= min_area) {
            int x = stat.at<int>(i, CC_STAT_LEFT);
            int y = stat.at<int>(i, CC_STAT_TOP);
            int w = stat.at<int>(i, CC_STAT_WIDTH);
            int h = stat.at<int>(i, CC_STAT_HEIGHT);
            ROIs.push_back(Rect(x,y,w,h));
        }
    }
    Scalar color= Scalar(0,255,0);
    int max_det = ROIs.size();
    Mat tmp;
    for(ulong i = 0; i < max_det; i++){
        img.copyTo(tmp);
        rectangle(tmp, ROIs[i], color);
        imshow("TMP", tmp);
        waitKey(0);
    }
    destroyAllWindows();
    return 0;
}
