//
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//void onMouse(int event, int x, int y, int flags, void* userdata) {
//    if (event == EVENT_LBUTTONDOWN) {
//        Mat img = *(Mat*)userdata;
//        Scalar intensity = img.at<Vec3b>(y, x);
//        cout << "Scalar value at (" << x << ", " << y << "): " << intensity << endl;
//    }
//}
//
//int main() {
//    Mat img = imread("D:\\×ÀÃæ\\test2.jpg");
//
//    if (img.empty()) {
//        cout << "Failed to load image" << endl;
//        return -1;
//    }
//
//    namedWindow("Image", WINDOW_NORMAL);
//    setMouseCallback("Image", onMouse, &img);
//
//    while (true) {
//        imshow("Image", img);
//        char c = waitKey(10);
//        if (c == 27) // Escape key
//            break;
//    }
//
//    return 0;
//}
