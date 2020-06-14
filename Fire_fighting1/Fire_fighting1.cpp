#include <iostream>  
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>  
#include<cmath>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
#include "cv.h"
using namespace cv;
using namespace std;
#define VIDEO_PATH		"E:\\last\\1.mp4"		
//火焰可利用亮度和颜色
//水柱的轨迹需要先用背景差分获得水柱的连通域，然后利用连通域上的像素点进行曲线的拟合
//水枪的位置视为已知，即可以手动活动坐标。

//曲线拟合函数，因上课无曲线拟合内容，参考CSDN博客： https://blog.csdn.net/guduruyu/article/details/72866144
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}

int main()
{
	VideoCapture capVideo(VIDEO_PATH);
	if (!capVideo.isOpened()) {//打开失败
		std::cout << "can not open" << std::endl;
		return -1;
	}

	int cnt = 0;//计数器
	Mat frame;
	Mat dst;
	Mat bgMat;
	Mat subMat;
	Mat bny_subMat;
	Mat bgModelMat;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	std::vector<cv::Point> points;
	cv::Mat A;

	while (1) {
		capVideo >> frame;
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		if (cnt == 0) {
			frame.copyTo(bgMat);//获得背景图像
			dst = frame.clone();//备份
		}
		else if (cnt == 100) {
			frame.copyTo(bgMat);//更新背景图像
			dst = frame.clone();//更新备份
		}
		else {//背景差分，第十二周内容
			absdiff(frame, bgMat, subMat);//背景图像和当前图像相减		
			threshold(subMat, bny_subMat, 20, 255, CV_THRESH_BINARY);//二值化
			//morphologyEx(bny_subMat, bny_subMat, MORPH_CLOSE, kernel); 
			morphologyEx(bny_subMat, bny_subMat, MORPH_OPEN, kernel);//开运算，第四周内容
			morphologyEx(bny_subMat, bny_subMat, MORPH_CLOSE, kernel);//闭运算，第四周内容
			//开始曲线拟合，因上课无曲线拟合内容，参考CSDN博客： https://blog.csdn.net/guduruyu/article/details/72866144
			
			//1.找点，已知水枪坐标（175，35）和火焰坐标
			if (cnt >= 100)
			{   //选择从水枪位置开始（175，35）到火焰左侧位置（370,162），减小误差。
				points.push_back(cv::Point(175., 35.));//水枪位置
				points.push_back(cv::Point(370., 162.));//火焰左侧位置
				points.push_back(cv::Point(422., 227.));//火焰位置
				points.push_back(cv::Point(425., 228.));//火焰位置
				points.push_back(cv::Point(428., 229.));//火焰位置
				points.push_back(cv::Point(430., 230.));//火焰位置
				points.push_back(cv::Point(432., 232.));//火焰位置
				for (int i = 170; i < 370; i++)	//遍历每列的误差小(从左到右)
				{
					for (int j = 35; j < 162; j++)	//行循环(从上往下遍历)
					{
						if (bny_subMat.at<uchar>(j, i) == 255)
						{
							points.push_back(cv::Point(i, j));
							break;
						}
					}
				}
			}

		
			//2.绘制折线
			if ((cnt >= 104)&& (cnt <= 145)) {
				polynomial_curve_fit(points, 3, A);
				std::cout << "A = " << A << std::endl;
				std::vector<cv::Point> points_fitted;
				for (int x = 170; x < 432; x++)//水枪到火焰
				{
					double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
						A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);
					points_fitted.push_back(cv::Point(x, y));
				}
				cv::polylines(dst, points_fitted, false, cv::Scalar(255, 0, 0), 1, 8, 0);

			}
			
	

			imshow("b_subMat", bny_subMat);
			imshow("frame", frame);
			imshow("dst", dst);
			waitKey(30);
		}
		cnt++;
	}
	return 0;
}
