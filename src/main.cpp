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
#include <utility>
#include <dirent.h>


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
vector<pair<string,string>> pairFiles(const vector<string>& image_names, const vector<string>& gt_names);
vector<pair<string,string>> getNames(const string& imagePath, const string& gtPath);

int main(int argc, char *argv[]) {

    //acquire paths from input
    if (argc != 3){
        cerr << "2 parameters allowed: images_path, ground_truth_path" << endl;
        exit(1);
    }
    string imgPath = string(argv[1]);
    string gtPath = string(argv[2]);

    // image full screen
    Display* disp = XOpenDisplay(nullptr);
    Screen*  scrn = DefaultScreenOfDisplay(disp);

    //read CNN
    Net net = readNetFromTensorflow("../../models/parts_sp_over_se_model.pb"); //Load the model
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);
    //read edge detector
    Ptr<StructuredEdgeDetection> edgeDetector = createStructuredEdgeDetection("../../EdgeModel/model.yml.gz");
    //build edgebox
    Ptr<EdgeBoxes> boxesDetector = createEdgeBoxes();

    //get pairs of image,ground truth files
    vector<pair<string,string>> names= getNames(imgPath, gtPath);

    for(auto& name : names) {

        //detect
        Mat img = imread(imgPath + name.first);
        vector<Rect> ROIs = detect(img, net, edgeDetector, boxesDetector);
        //read ground truth
        vector<Rect> groundTruth;
        if(!name.second.empty())
            groundTruth = parseFile(gtPath + name.second);

        //draw regions
        Scalar colorDec = Scalar(0, 0, 255); //red
        Scalar colorGT = Scalar(0, 255, 0); //green
        for (auto &ROI : ROIs) {
            rectangle(img, ROI, colorDec, 2);
        }
        //draw ground truth
        for (auto &ROI : groundTruth) {
            rectangle(img, ROI, colorGT, 2);
        }

        cout << name.first << " got an IoU score of " << IoUScore(groundTruth, ROIs) << endl;

        resize(img, img, Size(scrn->width - 100, scrn->height - 100));
        imshow("detection", img);
        waitKey(0);
    }

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

vector<pair<string,string>> pairFiles(const vector<string>& image_names, const vector<string>& gt_names){
    vector<pair<string,string>> out;

    for(auto& image : image_names){
        string img_raw = image.substr(0, image.find_last_of('.'));
        pair<string,string> p = pair<string,string>(image, string());
        for(auto& gt : gt_names){
            string gt_raw = gt.substr(0, gt.find_last_of('.'));
            if(img_raw == gt_raw) {
                p.second = gt;
                break;
            }
        }
        out.push_back(p);
    }
    return out;
}
vector<pair<string,string>> getNames(const string& imagePath, const string& gtPath){
    vector<string> img_names;
    vector<string> gt_names;
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir (imagePath.c_str())) != nullptr) {
        while ((ent = readdir (dir)) != nullptr)
            if (ent->d_type == DT_REG)
                img_names.emplace_back(ent->d_name);
        closedir (dir);
    }
    else {
        // could not open directory
        cerr << "could not open the images directory";
        exit(1);
    }

    if ((dir = opendir (gtPath.c_str())) != nullptr) {
        while ((ent = readdir (dir)) != nullptr)
            if (ent->d_type == DT_REG)
                gt_names.emplace_back(ent->d_name);
        closedir (dir);
    }
    else {
        // could not open directory
        cerr << "could not open the ground truth directory";
        exit(1);
    }

    return pairFiles(img_names, gt_names);
}