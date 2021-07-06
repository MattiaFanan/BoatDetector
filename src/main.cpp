#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("origin", img);
    Mat hsv_img;
    cvtColor(img,hsv_img,COLOR_BGR2HSV);
    Mat rows_ch = Mat::zeros(img.size[0], img.size[1], CV_32FC1);
    Mat cols_ch = Mat::zeros(img.size[0], img.size[1], CV_32FC1);

    for(int c=0; c < rows_ch.cols; c++)
        for(int r=0; r < rows_ch.rows; r++) {
            rows_ch.at<float>(r, c) = r ;
            cols_ch.at<float>(r, c) = c ;
        }

    Mat data;
    hsv_img.convertTo(data,CV_32F);

    vector<Mat> channels,tmp;
    split(data,tmp);
    for(auto &ch : tmp)
        channels.push_back(ch);
    channels.push_back(rows_ch);
    channels.push_back(cols_ch);
    merge(channels,data);

    cout << hsv_img.channels()<<endl;
    cout << data.channels()<<endl;

    data = data.reshape(1,data.rows * data.cols);

    int numLabels = 6;
    // do kmeans
    Mat labels, centers;
    kmeans(data, numLabels, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);

    // reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(5,centers.rows);
    data = data.reshape(5,data.rows);
    

    //assign colors to centers


    // replace pixel values with their center value:
    Vec<float,5> *p = data.ptr<Vec<float,5>>();
    for (size_t i=0; i<data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec<float,5>>(center_id);
    }

    // back to 2d, and uchar:
    img = data.reshape(3, img.rows);
    img.convertTo(img, CV_8U);

    imshow("segmented", img);
    waitKey(0);

    return 0;
}