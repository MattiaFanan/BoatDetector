#include <iostream>
#include <opencv2/saliency.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace saliency;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/02.png");

    HOGDescriptor hog = HOGDescriptor();
    hog.load("/home/mattia/Downloads/boat.xml");
    vector<Rect> ROIs;
    vector<double> weights;
    hog.detectMultiScale(img, ROIs,weights);

    Mat out;
    Scalar color = Scalar(0,0, 255);
    int max_det = ROIs.size();
    for(ulong i = 0; i < max_det; i++){
        img.copyTo(out);
        //putText(out, to_string(score[i]), Point(ROI[0]+20, ROI[1]+20), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color, 2, LINE_AA);
        rectangle(out, ROIs[i], color);
        imshow("TMP", out);
        waitKey(0);
    }
    return 0;
}
