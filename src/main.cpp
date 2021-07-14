#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/ximgproc/edgeboxes.hpp>
#include<opencv2/ximgproc.hpp>
#include <X11/Xlib.h>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ximgproc;
using namespace dnn;

vector<Rect> removeDuplicates(const vector<Rect> &input, double dimensionSlackPerc);
bool areSimilarRects(const Rect &r1, const Rect &r2, double dimensionSlackPerc);

int main() {
    Display* disp = XOpenDisplay(NULL);
    Screen*  scrn = DefaultScreenOfDisplay(disp);
    Net net = readNetFromTensorflow("/home/mattia/CLionProjects/CV/BoatDetector/models/sp_over_se_model.pb"); //Load the model

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);

    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image3161.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/10.jpg");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/07.png");
    resize(img,img,Size(scrn->width - 100,scrn->height - 100));

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

    //similar ROIs removal
    double dimensionSlackPerc = 0.1;
    ROIs = removeDuplicates(ROIs, dimensionSlackPerc);

    //classification
    vector<float> scores;
    for(Rect ROI : ROIs){
        Mat input = img(ROI);
        vector<Mat> out;
        input = blobFromImage(input,1.0/255,Size(100,100),Scalar(),false,false,CV_32F);
        net.setInput(input);
        net.forward(out);
        scores.push_back(out.at(0).at<float>(0,0));
    }

    // non maxima suppression
    float score_thresh = 0.8;
    float nms_thresh = 0.1;
    vector<int> keptIndices;
    NMSBoxes(ROIs,scores,score_thresh,nms_thresh,keptIndices);

    //draw regions
    Scalar color = Scalar(0,255,0);
    for(int i : keptIndices){
        putText(img, to_string(scores[i]),ROIs[i].tl() + Point(20,20),FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,color);
        rectangle(img, ROIs[i], color,2);
    }
    imshow("detection",img);
    waitKey(0);
    return 0;
}

vector<Rect> removeDuplicates(const vector<Rect> &input, double dimensionSlackPerc){
    ulong numElm = input.size();
    bool removed[input.size()];
    for(ulong i = 0; i < numElm; i++)
        removed[i] = false;

    for(ulong i = 0; i < numElm; i++) {
        if (!removed[i]) {
            for (ulong j = i + 1; j < numElm; j++) {
                if (!removed[j] && areSimilarRects(input[i], input[j], dimensionSlackPerc)) {
                    removed[j]=true;
                }
            }
        }
    }

    vector<Rect> output;
    for(ulong i = 0; i < numElm; i++) {
        if (!removed[i]) {
            output.push_back(input[i]);
        }
    }
    return output;

}

bool areSimilarRects(const Rect &r1, const Rect &r2, double dimensionSlackPerc){

    Point refTL = r1.tl();
    Point refBR = r1.br();

    Rect upBound = Rect(refTL*(1 - dimensionSlackPerc), refBR * (1 + dimensionSlackPerc));
    Rect lwBound = Rect(refTL*(1 + dimensionSlackPerc), refBR * (1 - dimensionSlackPerc));


    //if the corners are not inside the upper bound Rect then they are not similar
    if(!upBound.contains(r2.tl()) || !upBound.contains(r2.br()))
        return false;
    //now we are sure it is inside the upper bound
    //if the corners are inside the lower bound Rect then they are not similar
    if(lwBound.contains(r2.tl()) || lwBound.contains(r2.br()))
        return false;

    return true;
}
