////#include <opencv2/opencv.hpp>
////
////using namespace cv;
////
////int main()
////{
////    // 读取图像
////    Mat image = imread("D:\\桌面\\兑换站视频\\5.jpg");
////
////    // 将图像转换为灰度图像
////    Mat gray;
////    inRange(image, Scalar(105, 166, 230), Scalar(186, 253, 255), gray);
////
////    // 进行边缘检测
////    Mat edges;
////    Canny(gray, edges, 100, 200);
////
////    // 寻找轮廓
////    std::vector<std::vector<Point>> contours;
////    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
////
////    // 对每个轮廓进行处理
////    for (int i = 0; i < contours.size(); i++)
////    {
////        // 使用minEnclosingTriangle函数获取最小外接三角形
////        std::vector<Point2f> triangle;
////        minEnclosingTriangle(contours[i], triangle);
////
////        // 绘制轮廓和最小外接三角形
////        Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256); // 随机颜色
////        drawContours(image, contours, i, color, 2);
////        for (int j = 0; j < 3; j++)
////        {
////            line(image, triangle[j], triangle[(j + 1) % 3], color, 2);
////            // 计算直线角度
////            float angle = atan2(triangle[(j + 1) % 3].y - triangle[j].y, triangle[(j + 1) % 3].x - triangle[j].x) * 180 / CV_PI;
////            std::cout << "Triangle " << i << ", Edge " << j << ", Angle: " << angle << std::endl;
////        }
////    }
////
////
////    // 显示结果
////    imshow("Contours", image);
////    waitKey(0);
////    return 0;
////}
//
//// 三角形结构体，包括三条直线的倾角，三个点的坐标
//struct Triangle {
//    double angle1;
//    double angle2;
//    double angle3;
//    double edge_len1;
//    double edge_len2;
//    double edge_len3;
//    vector<Point2f> triangle_points;
//};
//
//// 找到最相似的三角形
//Triangle findMostSimilarTriangle(Triangle t1, Triangle t2, vector<Triangle> triangles) {
//    Triangle mostSimilar;
//    double minMatchValue = DBL_MAX;
//
//    for (auto triangle : triangles) {
//        double matchValue1 = matchShapes(t1.triangle_points, triangle.triangle_points, CV_CONTOURS_MATCH_I1, 0);
//        double matchValue2 = matchShapes(t2.triangle_points, triangle.triangle_points, CV_CONTOURS_MATCH_I1, 0);
//
//        // 相似度计算方式可以选择CV_CONTOURS_MATCH_I2或其他方式
//        double matchValue = matchValue1 + matchValue2; // 使用加和的方式综合两个三角形的相似度
//        if (matchValue < minMatchValue) {
//            minMatchValue = matchValue;
//            mostSimilar = triangle;
//        }
//    }
//
//    return mostSimilar;
//}
//
//
//void findMostSimilarContours(const vector<vector<Point> >& contours, const vector<vector<Point> >& knownContours) {
//
//    for (size_t i = 0; i < knownContours.size(); i++) {
//
//        double bestMatch = numeric_limits<double>::max();
//        int bestMatchIndex = -1;
//
//        for (size_t j = 0; j < contours.size(); j++) {
//
//            double match = matchShapes(knownContours[i], contours[j], CONTOURS_MATCH_I1, 0);
//            if (match < bestMatch) {
//                bestMatch = match;
//                bestMatchIndex = j;
//            }
//        }
//
//        cout << "Known contour " << i << " matches best with contour " << bestMatchIndex << " with match value " << bestMatch << endl;
//    }
//}