////#include <opencv2/opencv.hpp>
////
////using namespace cv;
////
////int main()
////{
////    // ��ȡͼ��
////    Mat image = imread("D:\\����\\�һ�վ��Ƶ\\5.jpg");
////
////    // ��ͼ��ת��Ϊ�Ҷ�ͼ��
////    Mat gray;
////    inRange(image, Scalar(105, 166, 230), Scalar(186, 253, 255), gray);
////
////    // ���б�Ե���
////    Mat edges;
////    Canny(gray, edges, 100, 200);
////
////    // Ѱ������
////    std::vector<std::vector<Point>> contours;
////    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
////
////    // ��ÿ���������д���
////    for (int i = 0; i < contours.size(); i++)
////    {
////        // ʹ��minEnclosingTriangle������ȡ��С���������
////        std::vector<Point2f> triangle;
////        minEnclosingTriangle(contours[i], triangle);
////
////        // ������������С���������
////        Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256); // �����ɫ
////        drawContours(image, contours, i, color, 2);
////        for (int j = 0; j < 3; j++)
////        {
////            line(image, triangle[j], triangle[(j + 1) % 3], color, 2);
////            // ����ֱ�߽Ƕ�
////            float angle = atan2(triangle[(j + 1) % 3].y - triangle[j].y, triangle[(j + 1) % 3].x - triangle[j].x) * 180 / CV_PI;
////            std::cout << "Triangle " << i << ", Edge " << j << ", Angle: " << angle << std::endl;
////        }
////    }
////
////
////    // ��ʾ���
////    imshow("Contours", image);
////    waitKey(0);
////    return 0;
////}
//
//// �����νṹ�壬��������ֱ�ߵ���ǣ������������
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
//// �ҵ������Ƶ�������
//Triangle findMostSimilarTriangle(Triangle t1, Triangle t2, vector<Triangle> triangles) {
//    Triangle mostSimilar;
//    double minMatchValue = DBL_MAX;
//
//    for (auto triangle : triangles) {
//        double matchValue1 = matchShapes(t1.triangle_points, triangle.triangle_points, CV_CONTOURS_MATCH_I1, 0);
//        double matchValue2 = matchShapes(t2.triangle_points, triangle.triangle_points, CV_CONTOURS_MATCH_I1, 0);
//
//        // ���ƶȼ��㷽ʽ����ѡ��CV_CONTOURS_MATCH_I2��������ʽ
//        double matchValue = matchValue1 + matchValue2; // ʹ�üӺ͵ķ�ʽ�ۺ����������ε����ƶ�
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