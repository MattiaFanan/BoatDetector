#include <iostream>
#include <opencv2/saliency.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace saliency;

int main() {
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    //cvtColor(img, img, COLOR_RGB2GRAY);
    /*
    Ptr<saliency::StaticSaliencyFineGrained> pointer = saliency::StaticSaliencyFineGrained::create();
    Mat out;
    Mat threshold_out;
    pointer->computeSaliency(img, out);
    out.convertTo(out, CV_8UC1, 255);
    threshold(out, threshold_out, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", threshold_out);
    waitKey(0);
     */
    String training_path = "/home/mattia/Dev/OpenCV_installation/opencv_contrib-master/modules/saliency/samples/ObjectnessTrainedModel";
    Ptr<Saliency> saliencyAlgorithm = ObjectnessBING::create();
    saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setTrainingPath( training_path );
    saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setBBResDir( "Results" );
    vector<Vec4i> saliencyMap;
    vector<Vec4i> ROIs;
    Mat out;
    img.copyTo(out);
    Scalar color = Scalar(0,0, 255);
    saliencyAlgorithm.dynamicCast<ObjectnessBING>()->computeSaliency(out, ROIs);

    vector<float> score = saliencyAlgorithm.dynamicCast<ObjectnessBING>()->getobjectnessValues();


    int max_det = 200;
    for(ulong i = 0; i < max_det; i++){
        Vec4i ROI = ROIs[i];
        img.copyTo(out);
        putText(out, to_string(score[i]), Point(ROI[0]+20, ROI[1]+20), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color, 2, LINE_AA);
        rectangle(out, Point(ROI[0], ROI[1]), Point(ROI[2], ROI[3]), color);
        imshow("TMP", out);
        waitKey(0);
    }

    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", out);
    waitKey(0);
    return 0;
}
