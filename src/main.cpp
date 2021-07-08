#include <iostream>
#include <opencv2/hfs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace hfs;
bool doOverlap(const Rect& rec1, const Rect& rec2, int slack);
int main() {
    namedWindow("merge",WINDOW_NORMAL);
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image3215.png");
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
    Mat tmp;
    /*
    for(auto &ROI : ROIs){
        img.copyTo(tmp);
        rectangle(tmp, ROI, color);
        imshow("TMP", tmp);
        waitKey(0);
    }
     */

    //merge ROIs
    Mat merg;
    int max_dist = 50;
    int index=81;
    Rect principal = ROIs[index];
    img.copyTo(merg);
    rectangle(merg, principal, color);
    for(int i=0; i<ROIs.size(); i++){
        if(i==index) continue;
        merg.copyTo(tmp);
        rectangle(tmp, ROIs[i], color);
        Rect ROI = ROIs[i];
        if(doOverlap(principal, ROI, max_dist)){
            Point newUpLeft = Point(min(principal.x, ROI.x), min(principal.y, ROI.y));
            Point newDownRight = Point(max(principal.x + principal.width, ROI.x + ROI.width),
                                       max(principal.y + principal.height, ROI.y + ROI.height));
            Rect newROI = Rect(newUpLeft,newDownRight);
            rectangle(tmp, newROI, Scalar(0,0,255));
        }
        imshow("merge", tmp);
        waitKey(0);
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}

bool doOverlap(const Rect& rec1, const Rect& rec2, int slack)
{
    Point l1 = Point(rec1.x, rec1.y);
    Point r1 = Point(rec1.x + rec1.width, rec1.y + rec1.height);
    Point l2 = Point(rec2.x, rec2.y);
    Point r2 = Point(rec2.x + rec2.width, rec2.y + rec2.height);

    //refuse if one into another
    if(rec1.contains(l2) && rec1.contains(r2) || rec2.contains(l1) && rec2.contains(r1))
        return false;

    // To check if either rectangle is actually a line
    // For example :  l1 ={-1,0}  r1={1,1}  l2={0,-1}
    // r2={0,1}
    if (l1.x == r1.x || l1.y == r1.y || l2.x == r2.x || l2.y == r2.y)
        // the line cannot have positive overlap
        return false;

    // If one rectangle is on left side of other
    if (l1.x - slack >= r2.x || l2.x >= r1.x + slack)
        return false;
    // If one rectangle is above other
    if (l1.y - slack >= r2.y || l2.y >= r1.y + slack)
        return false;

    return true;
}