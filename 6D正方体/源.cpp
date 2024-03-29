﻿#include"源.hpp"

std::ofstream myfile("D:\\桌面\\output.txt");
bool check_overlap(Mat& img, Triangle tri1, Triangle tri2)
{
	cout << tri1.triangle_points << endl << tri2.triangle_points << endl;
	cout << tri1.edge_len1 << "   " << tri1.edge_len2 << "   " << tri1.edge_len3 << "   " << tri2.edge_len1 << "   " << tri2.edge_len2 << "   " << tri2.edge_len3 << "   " << endl;
	cout << endl;
	cout << tri1.triangle_points.size() << tri2.triangle_points.size()<<endl<<endl;
    cout<<"uauaua"<<endl;
	if (tri1.edge_len1 <= 10 || tri1.edge_len2 <= 10 || tri1.edge_len3 <= 10 || tri2.edge_len1 <= 10 || tri2.edge_len2 <= 10 || tri2.edge_len3 <= 10)
	{
		cout << "ppppp" << endl;
		return 0;
	}
	cout << "dfaefasdfk" << endl;
	// 在两个空白图像中画出两个三角形
	Mat img1 = Mat::zeros(img.size(), CV_8UC1);
	Mat img2 = Mat::zeros(img.size(), CV_8UC1);
	if (tri1.triangle_points.size()==3 && tri2.triangle_points.size() == 3) 
	{
		vector<vector<Point>> triangles1 = { {tri1.triangle_points[0], tri1.triangle_points[1], tri1.triangle_points[2]} };
		vector<vector<Point>> triangles2 = { {tri2.triangle_points[0], tri2.triangle_points[1], tri2.triangle_points[2]} };
		cout << "lalala" << endl;
		fillPoly(img1, triangles1, Scalar(255));
		fillPoly(img2, triangles2, Scalar(255));
		cout << "bababa" << endl;
		// 检测两个三角形是否有重叠
		Mat result;
		bitwise_and(img1, img2, result);
		namedWindow("mask1", WINDOW_NORMAL);
		namedWindow("mask2", WINDOW_NORMAL);
		namedWindow("result", WINDOW_NORMAL);
		imshow("mask1", img1);
		imshow("mask2", img2);
		imshow("result", result);
		waitKey(1);
		cout << "mamama" << endl;
		if (countNonZero(result) == 0)
			return 0;
		else
			return 1;
	}
}

//图像预处理
Mat precessing(Mat image)
{
	// 转换为灰度图像
	Mat gray;
#if COLOR==1
	inRange(image, Scalar(230, 230, 230), Scalar(255, 255, 255), gray);
#else
	inRange(image, Scalar(9,32, 63), Scalar(60, 120, 255), gray);
#endif
	cv::namedWindow("gray", WINDOW_NORMAL);
	imshow("gray", gray);
	cv::waitKey(1);
	return gray;
}

//计算差异最小的三个数
void find_min_diff_indices(double arr[], int n, int& ind1, int& ind2, int& ind3) {
	double min_var = numeric_limits<double>::infinity();
	int idx1 = -1, idx2 = -1, idx3 = -1; // 方差最小的三个数的下标

	for (int i = 1; i < n - 1; i++) {
		double mean = (arr[i - 1] + arr[i] + arr[i + 1]) / 3.0;
		double variance = ((arr[i - 1] - mean) * (arr[i - 1] - mean) + (arr[i] - mean) * (arr[i] - mean) + (arr[i + 1] - mean) * (arr[i + 1] - mean))/(mean*mean*mean);

		if (variance < min_var) {
			min_var = variance;
			idx1 = i - 1;
			idx2 = i;
			idx3 = i + 1;
		}
	}

	//std::cout << "三个下标的数组值：" << endl;
	//std::cout << arr[idx1] << "  " << arr[idx2] << "   " << arr[idx3] << endl;
	//std::cout << "三个下标的值：" << endl;
	//std::cout << idx1 << "   " << idx2 << "   " << idx3 << endl;

	ind1 = idx1;
	ind2 = idx2;
	ind3 = idx3;
}

//返回最小差异角的平均值
double findMinangles(vector<double>& angles)
{
	double nums[10] = { 0 };
	int min1 = min((int)angles.size(), 10);
	for (int i = 0; i < min1; i++)
		nums[i] = angles[i];
	int n = min1;
	int idx1, idx2, idx3;
	find_min_diff_indices(nums, n, idx1, idx2, idx3);
	angles.erase(angles.begin() + idx1);
	angles.erase(angles.begin() + idx2);
	angles.erase(angles.begin() + idx3);
	double ans = (nums[idx1] + nums[idx2] + nums[idx3]) / 3.0;
	return ans;
}

//找四个顶点，返回四个顶点的vector
vector<Point2f> findApexs(vector<RotatedRect> minRect, vector<vector<Point>> contours,vector<Triangle> triangles, vector<int> index)
{
	//中心坐标计算
	float x = 0, y = 0;
	//std::cout << "findApexs中的index:";
	//for (int i = 0; i < index.size(); i++)
	//	std::cout << index[i];
	//std::cout << endl;
	for (int i = 0; i < 4; i++)
	{
		x += minRect[index[i]].center.x;
		y += minRect[index[i]].center.y;
	}
	x /= 4;
	y /= 4;//Point(x,y)是中心点
	std::cout <<"矩形中心点：" << Point(x, y) << endl;

	//确定四个点
	vector<Point2f> bigRect_points;
	for (int i = 0; i < index.size(); i++)
	{
		int max = 0, max_i = 0;
		float true_x = 0, true_y = 0;
		for (int j = 0; j < contours[index[i]].size(); j++)
		{
			float anotherx = contours[index[i]][j].x;
			float anothery = contours[index[i]][j].y;
			float differ = pow(x - anotherx, 2) + pow(y - anothery, 2);
			if (differ > max)
			{
				max = differ;
				true_x = anotherx;
				true_y = anothery;
			}
		}
		bigRect_points.push_back(Point2f(true_x, true_y));
	}
	return bigRect_points;
}

// 检测图像中的三角形并计算每个三角形的三条边的角度和长度
vector<Triangle> detectTriangles(const vector<vector<Point>>& contours, Mat& cap)
{
	vector<Triangle> triangles;

	// 对每个轮廓进行处理
	for (int i = 0; i < contours.size(); i++)
	{
		// 使用minEnclosingTriangle函数获取最小外接三角形
		vector<Point2f> triangle_points;
		minEnclosingTriangle(contours[i], triangle_points);

		// 计算每个三角形的三条边的角度
		Point2f pt1 = triangle_points[0];
		Point2f pt2 = triangle_points[1];
		Point2f pt3 = triangle_points[2];

		double angle1 = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180 / CV_PI;
		double angle2 = atan2(pt3.y - pt2.y, pt3.x - pt2.x) * 180 / CV_PI;
		double angle3 = atan2(pt1.y - pt3.y, pt1.x - pt3.x) * 180 / CV_PI;

		double len1 = sqrt(pow(pt2.y - pt1.y, 2) + pow(pt2.x - pt1.x, 2));
		double len2 = sqrt(pow(pt3.y - pt2.y, 2) + pow(pt3.x - pt2.x, 2));
		double len3 = sqrt(pow(pt1.y - pt3.y, 2) + pow(pt1.x - pt3.x, 2));

		Triangle t;
		t.angle1 = angle1;
		t.angle2 = angle2;
		t.angle3 = angle3;
		t.triangle_points = triangle_points;
		t.edge_len1 = len1;
		t.edge_len2 = len2;
		t.edge_len3 = len3;
		triangles.push_back(t);
	}

	return triangles;
}

//找第四个点，返回下标
int find_four_apex(vector<vector<Point>> contours,vector<Triangle> triangle, vector<int> findMinindex,Mat cap)
{
	if (findMinindex.size() != 3)
		std::cout << "error:find_four_apex" << endl;
	else
	{
		//	std::cout << "以下是三个三角形的所有信息：" << endl;
		//	// 遍历所有需要绘制的三角形
		//	for (int i = 0; i < findMinindex.size(); ++i) {
		//		std::cout << "三条边:" << endl;
		//		// 绘制当前三角形的颜色
		//		Scalar color(0, 255, 0);
		//		std::cout << triangle[findMinindex[i]].edge_len1 <<"   " << triangle[findMinindex[i]].edge_len2 <<"   " << triangle[findMinindex[i]].edge_len3 << endl;
		//		std::cout << "三个角：" << endl;
		//		std::cout << triangle[findMinindex[i]].angle1 << "   " << triangle[findMinindex[i]].angle2 << "   " << triangle[findMinindex[i]].angle3 << endl;
		//		// 在 cap 图像中绘制当前三角形
		//		for (int j = 0; j < 3; j++) {
		//			
		//			line(cap, triangle[findMinindex[i]].triangle_points[j], triangle[findMinindex[i]].triangle_points[(j + 1) % 3], color, 2);
		//		}
		//	}
		//	vector<double> handle_angles;//把三角形的三个角汇总
		//	for (int i = 0; i < findMinindex.size(); i++)
		//	{
		//		handle_angles.push_back(abs(triangle[findMinindex[i]].angle1));
		//		handle_angles.push_back(abs(triangle[findMinindex[i]].angle2));
		//		handle_angles.push_back(abs(triangle[findMinindex[i]].angle3));
		//	}
		//	std::cout << endl;
		//	sort(handle_angles.rbegin(), handle_angles.rend());
		//	std::cout << "三个三角形的三条边的夹角排序结果：" << endl;
		//	for (int i = 0; i < handle_angles.size(); i++)
		//		std::cout << handle_angles[i] << "   ";
		//	double angle1, angle2, angle3;
		//	angle1=findMinangles(handle_angles);
		//	angle2= findMinangles(handle_angles);
		//	std::cout <<"第一个平均值" << angle1 << " 第二个平均值  " << angle2 << endl;
		//	std::cout << endl << endl;
		//}

	}
	return 0;
}

//一个转换作用，输入面积序列，传出差异最小的三个面积下标
vector<int> findMinareas(vector<double> areas)//areas已经排序了(降序)
{
	double nums[7] = { 0 };
	int min1 = min((int)areas.size(), 7);
	for (int i = 0; i < min1; i++)
		nums[i] = areas[i];
	int n = min1;
	int idx1, idx2, idx3;
	find_min_diff_indices(nums, n, idx1, idx2, idx3);	

	vector<int> ans;
	ans.push_back(idx1);
	ans.push_back(idx2);
	ans.push_back(idx3);
	return ans;
}

//筛选出四个直角的灯条，返回下标
vector<int> handleLight(vector<vector<Point>> contours,vector<RotatedRect> minRect,vector<Triangle> triangle,Mat cap)
{
	//干扰项排除    现在的方案是1.去除太小的矩形2.找矩形面积最相似的三个面积
	vector<double> areas;
	for (int i = 0; i < minRect.size(); i++)
		if (minRect[i].size.area() > 50)
			areas.push_back(minRect[i].size.area());
	//findMinindex是下标
	vector<int> findMinindex = findMinareas(areas);
	//findMinindex.push_back((*max_element(findMinindex.begin(), findMinindex.end())) + 1);
	//找第四个顶点+转换参数
	vector<vector<Point>> tmp_contours;

	findMinindex.push_back(find_four_apex(tmp_contours,triangle,findMinindex,cap));
	//std::cout << "所有找到的斜矩形面积：" << endl;
	//for (int i = 0; i < minRect.size(); i++)
	//	std::cout << minRect[i].size.area() << "   ";
	//std::cout << endl << "所有三角形的信息：" << endl;
	//for (int i = 0; i < triangle.size(); i++)
	//	std::cout <<"第一个" << triangle[i].angle1 << " 第二个  " << triangle[i].angle2 << "  第三个  " << triangle[i].angle3 << "  三个点   " << triangle[i].triangle_points << endl;
	//std::cout << endl;
	return findMinindex;

}

//主函数，负责调用二值图后的全流程
vector<Point2f> handleMat(Mat src, Mat image)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// 找到所有轮廓
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
#ifdef GRAY
	std::cout << "轮廓数量：" << contours.size() << endl;
	if (contours.size() < 10000000000) {
		vector<Point2f> a;
		return a;
	}
#endif // GRAY
	// 为每个轮廓找到一个斜矩形和最小外接三角形
	vector<RotatedRect> minRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
		minRect[i] = minAreaRect(contours[i]);
	vector<Triangle> triangles = detectTriangles(contours, image);//调试，这个有用

	//将按照斜矩形的面积轮廓和斜矩形都排序，并保持轮廓和斜矩形变长数组的坐标一一对应
	vector<RotatedRect> handle_minRect(minRect.size()+5);
	vector<vector<Point>> handle_contours(contours.size()+5);
	vector<Triangle> handle_triangles(triangles.size()+5);
	//删除相交的轮廓，保留大的
	for (int i = 0; i < triangles.size()-1; i++) {
		cout << "大循环前的" << endl;
		for (int j = 0; j < triangles.size()-1; j++)
		{
			cout << "循环前的" << endl;
			if (check_overlap(image, triangles[i], triangles[j]))
			{
				myfile << "大函数里的"<<triangles[i].triangle_points << endl << triangles[j].triangle_points << endl << i << "   " << j << endl;
				if (minRect[i].size.area() > minRect[j].size.area()) {
					minRect[j] = RotatedRect(cv::Point2f(100, 100), cv::Size2f(0, 0), 0);
					contours[j].clear();
					triangles[j] = *new Triangle;
				}
				else {
					minRect[i] = RotatedRect(cv::Point2f(100, 100), cv::Size2f(0, 0), 0);
					contours[i].clear();
					triangles[i] = *new Triangle;
				}
			}
			cout << "循环后的" << endl;
			cout << triangles.size() << "   " << i << "    " << j << endl;
		}
		cout << "大循环后的" << endl;
		cout << triangles.size() << endl;
	}
	//保证斜矩形与轮廓的下标一样
	//自己写的选择排序(不想优化),按照minRect的面积大小排序
	for (int i = 0; i < contours.size(); i++) {
		int max = 0, max_i = 0;
		for (int j = 0; j < contours.size(); j++)
		{
			if (minRect[j].size.area() > max)
			{
				max = minRect[j].size.area();
				max_i = j;
			}
		}
		handle_minRect[i] = minRect[max_i];
		handle_contours[i] = contours[max_i];
		handle_triangles[i] = triangles[max_i];
		minRect[max_i] = RotatedRect(cv::Point2f(100, 100), cv::Size2f(0, 0), 0);
	}

	//判断至少有四个角
	vector<int> true_index;
	if (minRect.size() >= 4)
		true_index = handleLight(handle_contours,handle_minRect,handle_triangles,image);
	else {
		std::cout << "没有识别到四个角" << endl;
		vector<Point2f> a;
		return a;
	}

	//传参,找四个顶点
	vector<Point2f> true_vertexs = findApexs(handle_minRect, handle_contours,handle_triangles, true_index);

	Scalar colors[] = { Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 128, 255) };
	Mat cap1 = image.clone();
	Mat cap2 = image.clone();
	Mat cap3 = image.clone();
#ifdef DEBUG1//顶点
	// 定义颜色数组,RGB顺序分别是蓝，绿，红，亮蓝

	// 创建输出图像

	for (int i = 0; i < true_vertexs.size(); i++)
		circle(cap1, true_vertexs[i], 5, colors[i], -1);
	cv::namedWindow("cap1", WINDOW_NORMAL);
	cv::imshow("cap1", cap1);
#endif

#ifdef DEBUG2//斜矩形
	// 绘制旋转矩形
	for (int i = 0; i < true_index.size(); i++) {
		// 根据面积确定颜色
		Scalar color;
		color = colors[i];

		Point2f rect_points[4];
		handle_minRect[true_index[i]].points(rect_points);
		for (int j = 0; j < 4; j++) {
			line(cap2, rect_points[j], rect_points[(j + 1) % 4], color, 7, LINE_AA);
		}
	}
	cv::namedWindow("cap2", WINDOW_NORMAL);
	cv::imshow("cap2", cap2);
#endif // DEBUG2

#ifdef DEBUG3
	for (int i = 0; i < true_index.size(); i++) {
		Scalar color;
		color = colors[i];
		for (int j = 0; j < 3; j++)
		{
			line(cap3, handle_triangles[i].triangle_points[j], handle_triangles[i].triangle_points[(j + 1) % 3], color,7);
		}
	}
	cv::namedWindow("cap3", WINDOW_NORMAL);
	cv::imshow("cap3", cap3);
#endif
	waitKey(1);

	//返回值处理
	return true_vertexs;
}

//旋转向量转换成欧拉角
void rotationMatrixToEulerAngles(Mat& R, double& roll, double& pitch, double& yaw)
{

	double r11 = R.at<double>(0, 0);
	double r12 = R.at<double>(0, 1);
	double r13 = R.at<double>(0, 2);
	double r21 = R.at<double>(1, 0);
	double r22 = R.at<double>(1, 1);
	double r23 = R.at<double>(1, 2);
	double r31 = R.at<double>(2, 0);
	double r32 = R.at<double>(2, 1);
	double r33 = R.at<double>(2, 2);

	pitch = asin(-r31);

	if (cos(pitch) != 0) {
		roll = atan2(r32 / cos(pitch), r33 / cos(pitch));
		yaw = atan2(r21 / cos(pitch), r11 / cos(pitch));
	}
	else {
		roll = 0;
		yaw = atan2(-r12, r22);
	}
}

//pnp算法
void solveXYZ(vector<Point2f> vertices)
{
	double half_x;
	double half_y;
	double width_target;
	double height_target;

	double cam1[3][3] = {                                  //内参矩阵
		1689.2, 0, 624.7565,
		0, 1688.1, 496.4914,
		0, 0, 1 };

	double distCoeff1[5] = { 0.0163, -0.3351, 0, 0, 0 };   //畸变参数

	Mat cam_matrix = Mat(3, 3, CV_64FC1, cam1);
	Mat distortion_coeff = Mat(5, 1, CV_64FC1, distCoeff1);

	width_target = 24;             //长宽
	height_target = 24;

	std::vector<cv::Point2f> Points2D;    //图片坐标
	//Point2f vertices[4];
	Points2D.push_back(vertices[1]);
	Points2D.push_back(vertices[2]);
	Points2D.push_back(vertices[3]);
	Points2D.push_back(vertices[0]);

	std::vector<cv::Point3f> Point3d;     //世界坐标

	half_x = (width_target) / 2.0;
	half_y = (height_target) / 2.0;

	Point3d.push_back(Point3f(-half_x, half_y, 0));
	Point3d.push_back(Point3f(-half_x, -half_y, 0));
	Point3d.push_back(Point3f(half_x, -half_y, 0));
	Point3d.push_back(Point3f(half_x, half_y, 0));

	Mat rot1 = Mat::eye(3, 3, CV_64FC1);//旋转矩阵
	Mat trans1 = Mat::zeros(3, 1, CV_64FC1);//平移矩阵

	cv::solvePnP(Point3d, Points2D, cam_matrix, distortion_coeff, rot1, trans1, false);
	Mat_<double> rot_mat;
	cv::Rodrigues(rot1, rot_mat);

	double sy = std::sqrt(rot_mat.at<double>(0, 0) * rot_mat.at<double>(0, 0) + rot_mat.at<double>(1, 0) * rot_mat.at<double>(1, 0));

	bool singularot_mat = sy < 1e-6;

	double x, y, z;
	if (!singularot_mat)
	{
		x = std::atan2(rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
		y = std::atan2(-rot_mat.at<double>(2, 0), sy);
		z = std::atan2(rot_mat.at<double>(1, 0), rot_mat.at<double>(0, 0));
	}
	else
	{
		x = std::atan2(-rot_mat.at<double>(1, 2), rot_mat.at<double>(1, 1));
		y = std::atan2(-rot_mat.at<double>(2, 0), sy);
		z = 0;
	}

	// Convert angles to degrees
	x = x * 180.0 / CV_PI;
	y = y * 180.0 / CV_PI;
	z = z * 180.0 / CV_PI;
	/*cout << trans1 << endl;
	cout << "z轴转：" << x << "   x轴转：" << y << "   y轴转：" << z << endl;*/
	cout << trans1 << endl;
	cout << "z轴转：" << x << "    x轴转：" << y << "   y轴转：" << z << endl;
}

//提供给main的全调用函数
void all(Mat image)
{
	// 导入图像
	if (image.empty())
		return ;
	Mat processmat = precessing(image);//预处理完
	vector<Point2f> vertexs;//这是四个顶点
	vertexs = handleMat(processmat, image);//返回四个顶点
	if (vertexs.size() != 4)
		return;

	return ;
}
#ifdef ISMAP
int main()
{
	VideoCapture capture("D:\\桌面\\兑换站视频\\true_video.mp4");
	Mat image;
	while (1) {
		capture.read(image);
		if (!image.empty())
		{
			all(image);
			cv::namedWindow("image", WINDOW_NORMAL);
			imshow("image", image);
			std::cout << endl << endl << endl;
			int c = waitKey(1);
			if (c == 27)
				break;
		}
	}

}

#else
int main()
{
	// 导入图像
	Mat image = imread("D:\\桌面\\兑换站视频\\5.jpg");
	Mat processmat = precessing(image);
	vector<Point2f> vertexs;//这是四个顶点
	vertexs = handleMat(processmat, image);
	cout << "四个点：" << endl;
	for (int i = 0; i < vertexs.size(); i++)
		cout << vertexs[i] << endl;
	solveXYZ(vertexs);
	waitKey(0);
	return 0;
}
#endif // ISMAP 1}