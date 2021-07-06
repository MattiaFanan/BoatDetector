#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace ximgproc::segmentation;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("origin", img);

    Mat out;
    vector<Rect> ROIs;
    Scalar color = Scalar(0,255,0);
    double sigma=0.5;
    float k=300;
    int min_size=100;
    Ptr<SelectiveSearchSegmentation> segmenter = createSelectiveSearchSegmentation();
    segmenter->clearStrategies();
    segmenter->addStrategy(createSelectiveSearchSegmentationStrategyTexture());
    segmenter->setBaseImage(img);
    segmenter->switchToSingleStrategy();

    segmenter->process(ROIs);

    int max_det = 200;
    for(ulong i = 0; i < max_det; i++){
        img.copyTo(out);
        rectangle(out,ROIs[i],color,2);
        imshow("TMP", out);
        waitKey(0);
    }

    return 0;
}