#include <iostream>
#include <opencv2/saliency.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace saliency;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    Ptr<saliency::StaticSaliencySpectralResidual> pointer = saliency::StaticSaliencySpectralResidual::create();


    /*cvtColor(img, img, COLOR_RGB2GRAY);
    Mat out;
    Mat threshold_out;
    pointer->computeSaliency(img, out);
    imshow("saliency", out);
    waitKey(0);
    out.convertTo(out, CV_8UC1, 255);
    threshold(out, threshold_out, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("TMP", img);
    waitKey(0);
    imshow("TMP", threshold_out);
    waitKey(0);*/

    Mat brg[3];
    Mat brg_sal[3];
    split(img,brg);

    for( int i = 0 ; i<3 ; i++) {
        pointer->computeSaliency(brg[i], brg_sal[i]);
        brg_sal[i].convertTo(brg_sal[i], CV_8UC1, 255);
        threshold(brg_sal[i], brg_sal[i], 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    Mat or_img;
    bitwise_or(brg_sal[0],brg_sal[1],or_img);
    bitwise_or(or_img,brg_sal[2],or_img);

    imshow("orig",img);
    //imshow("b",brg_sal[0]);
    //imshow("r",brg_sal[1]);
    //imshow("g",brg_sal[2]);
    imshow("or",or_img);

    Mat g;
    pointer->computeSaliency(img, g);
    g.convertTo(g, CV_8UC1, 255);
    threshold(g, g, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("img",g);

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    Mat cla_img;
    cvtColor(img,cla_img,COLOR_RGB2GRAY);
    clahe->apply(cla_img,cla_img);
    pointer->computeSaliency(cla_img, cla_img);
    g.convertTo(cla_img, CV_8UC1, 255);
    threshold(cla_img, cla_img, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("CLAHE",cla_img);


    waitKey(0);



    /*
    int erosion_size = 1;
    Mat element = getStructuringElement( MORPH_RECT,Size( 2*erosion_size + 1, 2*erosion_size+1 ));
    morphologyEx( threshold_out, threshold_out, MORPH_DILATE, element );
    //erode(threshold_out,threshold_out,element);*/

    /*
    Mat labelImage, stat, centroid;
    int nLabels = connectedComponentsWithStats(threshold_out, labelImage, stat,centroid,8);

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
    waitKey(0);
    */
    return 0;
}
