#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <climits>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include<fstream>
#include <numeric>
#define DEBUG1                         //画顶点
#define DEBUG2                       //画斜矩形
#define DEBUG3                       //画三角形
#define COLOR 0//0红色，1蓝色
#define ISMAP//1是图片，2是视频
using namespace std;
using namespace cv;

// 三角形结构体，包括三条直线的倾角，三个点的坐标
struct Triangle {
	double angle1 = 0;
	double angle2 = 0;
	double angle3 = 0;
	double edge_len1 = 0;
	double edge_len2 = 0;
	double edge_len3 = 0;
	vector<Point2f> triangle_points;
};

bool check_overlap(Mat& img, Triangle tri1, Triangle tri2);

//double类型最大值
const double INF = numeric_limits<double>::infinity();

Mat precessing(Mat image);

void find_min_diff_indices(double arr[], int n, int& ind1, int& ind2, int& ind3);

double findMinangles(vector<double>& angles);

vector<Point2f> findApexs(vector<RotatedRect> minRect, vector<vector<Point>> contours, vector<Triangle> triangles, vector<int> index);

vector<Triangle> detectTriangles(const vector<vector<Point>>& contours, Mat& cap);

int find_four_apex(vector<vector<Point>> contours, vector<Triangle> triangle, vector<int> findMinindex, Mat cap);

vector<int> findMinareas(vector<double> areas);

vector<int> handleLight(vector<vector<Point>> contours, vector<RotatedRect> minRect, vector<Triangle> triangle, Mat cap);

vector<Point2f> handleMat(Mat src, Mat image);

void rotationMatrixToEulerAngles(Mat& R, double& roll, double& pitch, double& yaw);

void solveXYZ(vector<Point2f> vertices);

void all(Mat image);