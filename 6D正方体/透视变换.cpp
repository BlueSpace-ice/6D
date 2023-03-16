//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char** argv)
//{
//    // 读入原始图片
//    Mat src = imread("D:\\桌面\\兑换站视频\\5.jpg");
//    if (src.empty()) {
//        cout << "Could not open or find the image!\n";
//        return -1;
//    }
//
//    // 定义原始图片四个顶点及其对应的目标位置
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
//    // 计算透视变换矩阵
//    Mat perspectiveMatrix = getPerspectiveTransform(srcPoints, dstPoints);
//
//    // 应用透视变换
//    Mat dst;
//    warpPerspective(src, dst, perspectiveMatrix, src.size());
//
//    // 显示原始图片和变换后的图片
//    namedWindow("Original Image", WINDOW_NORMAL);
//    namedWindow("Perspective Transformation", WINDOW_NORMAL);
//    imshow("Original Image", src);
//    imshow("Perspective Transformation", dst);
//
//    waitKey(0);
//    return 0;
//}



