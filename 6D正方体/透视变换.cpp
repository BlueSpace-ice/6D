//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char** argv)
//{
//    // ����ԭʼͼƬ
//    Mat src = imread("D:\\����\\�һ�վ��Ƶ\\5.jpg");
//    if (src.empty()) {
//        cout << "Could not open or find the image!\n";
//        return -1;
//    }
//
//    // ����ԭʼͼƬ�ĸ����㼰���Ӧ��Ŀ��λ��
//    vector<Point2f> srcPoints(4);
//    vector<Point2f> dstPoints(4);
//    srcPoints[0] = Point2f(0, 0);
//    srcPoints[1] = Point2f(src.cols - 1, 0);
//    srcPoints[2] = Point2f(src.cols - 1, src.rows - 1);
//    srcPoints[3] = Point2f(0, src.rows - 1);
//    dstPoints[0] = Point2f(src.cols * 0.1, src.rows * 0.2);
//    dstPoints[1] = Point2f(src.cols * 0.9, src.rows * 0.1);
//    dstPoints[2] = Point2f(src.cols * 0.9, src.rows * 0.9);
//    dstPoints[3] = Point2f(src.cols * 0.1, src.rows * 0.8);
//
//
//    // ����͸�ӱ任����
//    Mat perspectiveMatrix = getPerspectiveTransform(srcPoints, dstPoints);
//
//    // Ӧ��͸�ӱ任
//    Mat dst;
//    warpPerspective(src, dst, perspectiveMatrix, src.size());
//
//    // ��ʾԭʼͼƬ�ͱ任���ͼƬ
//    namedWindow("Original Image", WINDOW_NORMAL);
//    namedWindow("Perspective Transformation", WINDOW_NORMAL);
//    imshow("Original Image", src);
//    imshow("Perspective Transformation", dst);
//
//    waitKey(0);
//    return 0;
//}



