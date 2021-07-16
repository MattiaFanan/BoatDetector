#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/ximgproc/edgeboxes.hpp>
#include<opencv2/ximgproc.hpp>
#include <X11/Xlib.h>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <Hungarian.h>


using namespace cv;
using namespace std;
using namespace ximgproc;
using namespace dnn;

vector<Rect> removeDuplicates(const vector<Rect> &input, double dimensionSlackPerc);
bool areSimilarRects(const Rect &r1, const Rect &r2, double dimensionSlackPerc);
vector<Rect> parseFile(const string& fileName);
double IoU(const Rect &r1, const Rect &r2);
double IoUScore(const vector<Rect> &groundTruth, const vector<Rect> &detection);
vector<Rect> detect(const Mat& img, Net net, const Ptr<StructuredEdgeDetection>& edgeDetector, const Ptr<EdgeBoxes>& boxesDetector);

int main() {
    // image full screen
    Display* disp = XOpenDisplay(NULL);
    Screen*  scrn = DefaultScreenOfDisplay(disp);

    //read CNN
    Net net = readNetFromTensorflow("/home/mattia/CLionProjects/CV/BoatDetector/models/parts_sp_over_se_model.pb"); //Load the model
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);

    //read edge detector
    Ptr<StructuredEdgeDetection> edgeDetector = createStructuredEdgeDetection("/home/mattia/Downloads/model.yml.gz");

    //build edgebox
    Ptr<EdgeBoxes> boxesDetector = createEdgeBoxes();

    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0081.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/07.jpg");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/11.png");



    //read ground truth
    string fileName = string("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/LABELS_TXT/image0081.txt");
    //string fileName = string("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle_labels_txt/07.txt");
    //string fileName = string("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice_labels_txt/07.txt");
    vector<Rect> groundTruth = parseFile(fileName);

    //detect
    vector<Rect> ROIs = detect(img,net,edgeDetector,boxesDetector);


    //draw regions
    Scalar colorDec = Scalar(0,0,255); //red
    Scalar colorGT = Scalar(0,255,0); //green

    for(auto& ROI : ROIs){
        rectangle(img, ROI, colorDec, 2);
    }

    //draw ground truth
    for(auto &ROI : groundTruth){
        rectangle(img, ROI, colorGT, 2);
    }

    cout << IoUScore(groundTruth,ROIs) << endl;

    resize(img,img,Size(scrn->width - 100,scrn->height - 100));
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

vector<Rect> parseFile(const string& fileName){
    vector<Rect> ROIs;
    ifstream inFile;
    inFile.open(fileName);
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    string textRow;
    regex e(R"(\d+)");
    while(inFile >> textRow) {
        sregex_iterator iter = sregex_iterator(textRow.cbegin(), textRow.cend(), e);
        sregex_iterator end;

        vector<int> num;
        for (; iter != end; ++iter)
            num.push_back(stoi(iter->str()));
        if(num.size()!=4) {
            cerr <<"error parsing ROIs file: one ROI hasn't exactly 4 integers";
            exit(1);
        }
        //0 xmin; 1 xmax; 2 ymin; 3 ymax
        ROIs.emplace_back(num[0],num[2], num[1] - num[0] + 1,num[3] - num[2] + 1);

    }
    inFile.close();
    return ROIs;
}

double IoU(const Rect &r1, const Rect &r2){
    //determine the (x, y)-coordinates of the intersection rectangle
    int xA = max(r1.tl().x, r2.tl().x);
    int yA = max(r1.tl().y, r2.tl().y);
    int xB = min(r1.br().x, r2.br().x);
    int yB = min(r1.br().y, r2.br().y);
    //compute the area of intersection rectangle
    int interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);

    return interArea / ( r1.area() + r2.area() - interArea + 1.0e-16);
}

double IoUScore(const vector<Rect> &groundTruth, const vector<Rect> &detection) {
    //Mat pairIoU = Mat(static_cast<int>(groundTruth.size()), static_cast<int>(detection.size()), CV_32F);
    if (groundTruth.empty())
        return 0;

    int dim = static_cast<int>(max(groundTruth.size(), detection.size()));

    vector<vector<double>> costs;
    for (int i = 0; i < dim; i++) {
        vector<double> gtCosts;
        for (int j = 0; j < dim; j++) {
            if (i < groundTruth.size() && j < detection.size())
                gtCosts.push_back(1.0 - IoU(groundTruth[i], detection[j]));
            else
                gtCosts.push_back(1.0);
        }
        costs.push_back(gtCosts);
    }

    vector<int> assignments;
    HungarianAlgorithm HungAlgo;
    double sum = HungAlgo.Solve(costs, assignments);
    sum = 1 - sum / dim;
    return sum;
}


vector<Rect> detect(const Mat& img, Net net, const Ptr<StructuredEdgeDetection>& edgeDetector, const Ptr<EdgeBoxes>& boxesDetector){
    vector<Rect> ROIs;
    Mat tmp;
    //normalize and copy img
    img.convertTo(tmp,CV_32FC3,1/255.0);

    //#//find regions of interest
    Mat edges,orient,suppEdges;
    edgeDetector->detectEdges(tmp,edges);
    edgeDetector->computeOrientation(edges,orient);
    //edge suppression
    edgeDetector->edgesNms(edges,orient,suppEdges);
    // save ROIs
    boxesDetector->getBoundingBoxes(suppEdges,orient,ROIs);

    //similar ROIs removal
    double dimensionSlackPerc = 0.05;
    ROIs = removeDuplicates(ROIs, dimensionSlackPerc);

    //classification
    vector<float> scores;
    for(const Rect& ROI : ROIs){
        Mat input = tmp(ROI);
        vector<Mat> out;
        input = blobFromImage(input,1.0,Size(100,100),Scalar(),false,false,CV_32F);
        net.setInput(input);
        net.forward(out);
        scores.push_back(out.at(0).at<float>(0,0));
    }

    // non maxima suppression
    float score_thresh = 0.8;
    float nms_thresh = 0.1;
    vector<int> keptIndices;
    NMSBoxes(ROIs,scores,score_thresh,nms_thresh,keptIndices);

    vector<Rect> output;
    for(int i : keptIndices)
        output.push_back(ROIs[i]);

    return output;
}