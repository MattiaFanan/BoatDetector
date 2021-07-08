#include <iostream>
#include <opencv2/hfs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <X11/Xlib.h>

using namespace cv;
using namespace std;
using namespace hfs;

bool doOverlap(const Rect& rec1, const Rect& rec2, int slack);
vector<vector<Rect>> proposedRegions(const vector<Rect> &zeroLevel, int max_dist, int maxLevel);
vector<Rect> ROIsFromStat(const Mat &stat, int nLabels, int minArea);

int main() {
    Display* disp = XOpenDisplay(NULL);
    Screen*  scrn = DefaultScreenOfDisplay(disp);

    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image3215.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    resize(img,img,Size(scrn->width - 100,scrn->height - 100));
    imshow("orig",img);
    waitKey(0);

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
    int minArea = 200;
    int maxDist = 50;
    int maxLevel = 4;
    vector<Rect> zeroLevel = ROIsFromStat(stat, nLabels, minArea);
    vector<vector<Rect>> levelStack = proposedRegions(zeroLevel, maxDist, maxLevel);
    Scalar color= Scalar(0,0,255);
    Mat tmp;
    int level = 0;
    for(auto &ROIs : levelStack){
        for(auto &ROI : ROIs) {
            img.copyTo(tmp);
            putText(tmp, to_string(level), Point(50,50),FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,color,2);
            rectangle(tmp, ROI, color,2);
            imshow("ROIs", tmp);
            waitKey(0);
        }
        level++;
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

vector<vector<Rect>> proposedRegions(const vector<Rect> &zeroLevel, int max_dist, int maxLevel){
    vector<vector<Rect>> levelStack;
    levelStack.push_back(zeroLevel);

    for(int toMergeLevel=0; toMergeLevel < maxLevel; toMergeLevel++){
        vector<Rect> newLevel;
        vector<Rect> currentLevel = levelStack[toMergeLevel];
        for(int i=0; i < currentLevel.size(); i++) {
            Rect pivotROI = currentLevel[i];
            for(int j= i + 1; j < currentLevel.size(); j++){
                Rect currentROI = currentLevel[j];
                if (doOverlap(pivotROI, currentROI, max_dist)) {
                    //the rect union of two rect has as top left the min of the top left points
                    //and as bottom right the max of the bottom right points
                    Point newUpLeft = Point(min(pivotROI.x, currentROI.x), min(pivotROI.y, currentROI.y));
                    Point newDownRight = Point(max(pivotROI.x + pivotROI.width, currentROI.x + currentROI.width),
                                           max(pivotROI.y + pivotROI.height, currentROI.y + currentROI.height));
                    //add the new ROI
                    Rect newROI = Rect(newUpLeft, newDownRight);
                    newLevel.push_back(newROI);
                }
            }
        }
        //exit if no new levels generated
        if(newLevel.empty())
            break;

        levelStack.push_back(newLevel);
    }
    return levelStack;
}

vector<Rect> ROIsFromStat(const Mat &stat, int nLabels, int minArea){
    vector<Rect> ROIs;
    for(int i=0; i < nLabels; i++){
        if(stat.at<int>(i,CC_STAT_AREA) >= minArea) {
            int x = stat.at<int>(i, CC_STAT_LEFT);
            int y = stat.at<int>(i, CC_STAT_TOP);
            int w = stat.at<int>(i, CC_STAT_WIDTH);
            int h = stat.at<int>(i, CC_STAT_HEIGHT);
            ROIs.push_back(Rect(x,y,w,h));
        }
    }
    return ROIs;
}