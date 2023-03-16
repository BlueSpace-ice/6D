//#include<iostream>
//#include<opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//class Armor //װ�װ���
//{
//public:
//	RotatedRect RRect;
//	Rect rect;                     //�������Σ�װ�װ�
//	Point center;                  //װ�װ����ĵ�
//
//	bool Fire = true;              //һ������ƥ�䵽����װ�װ壬ȡ��ƽ�е�
//	cv::Point armor_four_point[6]; //װ�װ���ĵ�
//
//	float ParallelAngel;
//
//	float distance; //��ľ���
//};
//
//void solveXYZ(Armor FinallyArmor, Point3f& poi3)
//{
//	double width_target;
//	double height_target;
//
//	width_target = 10;
//	height_target = 10;
//	double cam1[3][3] = {
//		1689.2, 0, 624.7565,
//		0, 1688.1, 496.4914,
//		0, 0, 1 };
//
//	double distCoeff1[5] = { 0.0163, -0.3351, 0, 0, 0 };   //�������
//
//
//
//	Mat cam_matrix = Mat(3, 3, CV_64FC1, cam1);
//	Mat distortion_coeff = Mat(5, 1, CV_64FC1, distCoeff1);
//
//	std::vector<cv::Point2f> Points2D;
//	Point2f vertices[4];
//	FinallyArmor.RRect.points(vertices);
//
//	Points2D.push_back(vertices[1]);
//	Points2D.push_back(vertices[2]);
//	Points2D.push_back(vertices[3]);
//	Points2D.push_back(vertices[0]);
//
//	std::vector<cv::Point3f> Point3d;
//	double half_x = width_target / 2.0;
//	double half_y = height_target / 2.0;
//
//	Point3d.push_back(Point3f(-half_x, -half_y, 0));
//	Point3d.push_back(Point3f(half_x, -half_y, 0));
//	Point3d.push_back(Point3f(half_x, half_y, 0));
//	Point3d.push_back(Point3f(-half_x, half_y, 0));
//
//	Mat rot = Mat::eye(3, 3, CV_64FC1);
//	Mat trans = Mat::zeros(3, 1, CV_64FC1);
//
//	cv::solvePnP(Point3d, Points2D, cam_matrix, distortion_coeff, rot, trans, false); //Gao�ķ�������ʹ�������ĸ������㣬������������������4Ҳ���ܶ���4
//
//	double x = trans.at<double>(0, 0);
//	double y = trans.at<double>(1, 0);
//	double z = trans.at<double>(2, 0);
//	double dist = sqrt(trans.at<double>(0, 0) * trans.at<double>(0, 0) + trans.at<double>(1, 0) * trans.at<double>(1, 0) + trans.at<double>(2, 0) * trans.at<double>(2, 0));
//
//	Mat rot1 = Mat::eye(3, 3, CV_64FC1);
//	Mat trans1 = Mat::zeros(3, 1, CV_64FC1);
//
//	cv::solvePnP(Point3d, Points2D, cam_matrix, distortion_coeff, rot1, trans1, false);///*�ڶ��ε���solvePnP����δ��*///
//
//	double x1 = trans1.at<double>(0, 0);
//	double y1 = trans1.at<double>(1, 0);
//	double z1 = trans1.at<double>(2, 0);
//	double dist1 = sqrt(trans1.at<double>(0, 0) * trans1.at<double>(0, 0) + trans1.at<double>(1, 0) * trans1.at<double>(1, 0) + trans1.at<double>(2, 0) * trans1.at<double>(2, 0));
//
//	float pit_angle = -atan(y1 / z1) * 180 / CV_PI;
//	float yaw_angle = -atan(x1 / z1) * 180 / CV_PI;
//
//	cout << "PNP�ĽǶ�p1y1d1===  " << pit_angle << "   " << yaw_angle << "   " << dist1 << endl;
//	cout << rot1.at<double>(0) / CV_PI * 180 << endl;
//	cout << rot1.at<double>(1) / CV_PI * 180 << endl;
//	cout << rot1.at<double>(2) / CV_PI * 180 << endl;
//
//	poi3 = Point3f(x1, y1, z1);
//}
//
//int main()
//{
//
//}