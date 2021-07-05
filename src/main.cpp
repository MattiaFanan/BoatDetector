#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>

using namespace cv;
using namespace std;

int main() {
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TRAINING_DATASET/IMAGES/image0147.png");
    Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/venice/05.png");
    //Mat img = imread("/home/mattia/CLionProjects/CV/BoatDetector/FINAL_DATASET/TEST_DATASET/kaggle/01.jpg");
    imshow("origin", img);
    // convert to float & reshape to a [3 x W*H] Mat
    //  (so every pixel is on a row of it's own)
    Mat data;
    img.convertTo(data,CV_32F);
    data = data.reshape(1,data.total());
    int numLabels = 8;
    // do kmeans
    Mat labels, centers;
    kmeans(data, numLabels, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);

    // reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(3,centers.rows);
    data = data.reshape(3,data.rows);

    // replace pixel values with their center value:
    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i=0; i<data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

    // back to 2d, and uchar:
    img = data.reshape(3, img.rows);
    img.convertTo(img, CV_8U);

    imshow("segmented", img);
    waitKey(0);

    return 0;
}