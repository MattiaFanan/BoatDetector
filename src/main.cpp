#include <iostream>
#include <opencv2/saliency.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/ximgproc/edgeboxes.hpp>
#include<opencv2/ximgproc.hpp>
#include <X11/Xlib.h>

using namespace cv;
using namespace std;
using namespace ximgproc;



int main() {
    Display* disp = XOpenDisplay(NULL);
    Screen*  scrn = DefaultScreenOfDisplay(disp);

    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image3161.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/10.jpg");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/01.png");
    resize(img,img,Size(scrn->width - 100,scrn->height - 100));
    imshow("orig",img);


    Mat tmp;
    img.convertTo(tmp,CV_32FC3,1/255.0);
    Mat edges,orient,suppEdges;
    Ptr<StructuredEdgeDetection> edgeDetector = createStructuredEdgeDetection("/home/mattia/Downloads/model.yml.gz");
    edgeDetector->detectEdges(tmp,edges);
    edgeDetector->computeOrientation(edges,orient);
    //edge suppression
    edgeDetector->edgesNms(edges,orient,suppEdges);

    vector<Rect> ROIs;
    Ptr<EdgeBoxes> boxes = createEdgeBoxes();
    //boxes->setMaxBoxes(30);
    boxes->getBoundingBoxes(suppEdges,orient,ROIs);

    Scalar color = Scalar(0,255,0);
    int max_det = ROIs.size();
    for(ulong i = 0; i < max_det; i++){
        img.copyTo(tmp);
        //putText(out, to_string(score[i]), Point(ROI[0]+20, ROI[1]+20), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color, 2, LINE_AA);
        rectangle(tmp, ROIs[i], color,2);
        imshow("TMP", tmp);
        waitKey(0);
    }

    waitKey(0);
    return 0;
}
