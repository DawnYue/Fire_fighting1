//
//  main.cpp
//  test
//
//  Created by 徐亦燊 on 2020/2/28.
//  Copyright © 2020 徐亦燊. All rights reserved.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

//识别并标出火焰
void fire(Mat& srcMat, Mat& outputMat);
bool curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

int main()
{
	VideoCapture capVideo("E://last/1.mp4");

	int cnt = 0;//计数器
	Mat frame;
	Mat dst;
	Mat dst1;
	Mat bgMat;
	Mat subMat;
	Mat bny_subMat;
	Mat bgModelMat;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	std::vector<cv::Point> points;
	cv::Mat A;

	while (1) {
		capVideo >> frame;
		if (!frame.data)
		{
			cout << "src image load failed!" << endl;
			break;
		}

		dst = frame.clone();
		dst1 = frame.clone();//保存原图像


		cvtColor(frame, frame, COLOR_BGR2GRAY);


		if (cnt == 0) {
			frame.copyTo(bgMat);//获得背景图像
			fire(dst, dst);//找火
			imshow("output", dst);
		}
		else if (cnt <= 90) {
			frame.copyTo(bgMat);//更新背景图像
			fire(dst, dst);//找火
			imshow("output", dst);
		}
		else {//背景差分，第十二周内容
			absdiff(frame, bgMat, subMat);//背景图像和当前图像相减
			threshold(subMat, bny_subMat, 20, 255, THRESH_BINARY);//二值化
			//morphologyEx(bny_subMat, bny_subMat, MORPH_CLOSE, kernel);
			morphologyEx(bny_subMat, bny_subMat, MORPH_OPEN, kernel);//开运算，第四周内容
			morphologyEx(bny_subMat, bny_subMat, MORPH_CLOSE, kernel);//闭运算，第四周内容
			//开始曲线拟合，因上课无曲线拟合内容，参考CSDN博客： https://blog.csdn.net/guduruyu/article/details/72866144

			fire(dst, dst);//找火
			imshow("output", dst);
			//1.找点，已知水枪坐标（175，35）和火焰坐标
			//选择从水枪位置开始（175，35）到火焰左侧位置（370,162），减小误差。
			points.push_back(cv::Point(175., 35.));//水枪位置
			points.push_back(cv::Point(370., 162.));//火焰左侧位置
			points.push_back(cv::Point(422., 227.));//火焰位置
			points.push_back(cv::Point(425., 228.));//火焰位置
			points.push_back(cv::Point(428., 229.));//火焰位置
			points.push_back(cv::Point(430., 230.));//火焰位置
			points.push_back(cv::Point(432., 232.));//火焰位置
			for (int i = 230; i < 370; i++)    //遍历每列的误差小(从左到右)
			{
				for (int j = 35; j < 162; j++)    //行循环(从上往下遍历)
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
		curve_fit(points, 3, A);
		std::cout << "A = " << A << std::endl;
		std::vector<cv::Point> points_fitted;
		for (int x = 170; x < 432; x++)//水枪到火焰
		{
			double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
				A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);
			points_fitted.push_back(cv::Point(x, y));
		}
		cv::polylines(dst, points_fitted, false, cv::Scalar(255, 0, 0), 1, 8, 0);

		fire(dst1, dst);
		imshow("frame", frame);
		imshow("dst", dst);
		waitKey(30);
		cnt++;
	}
	return 0;
}


//识别并标出火焰
//输入 原图像 输出 红线标出火焰轮廓
void fire(Mat& srcMat, Mat& outputMat)
{
	Mat gray;
	Mat hsvMat;
	cvtColor(srcMat, gray, COLOR_BGR2GRAY);
	Mat dstMat = Mat::zeros(gray.size(), gray.type());
	cvtColor(srcMat, hsvMat, COLOR_BGR2HSV);
	vector<Mat> channels;
	//分离通道
	split(hsvMat, channels);

	//Mat H_Mat;
	Mat S_Mat;
	//Mat V_Mat;
	//channels.at(0).copyTo(H_Mat);
	channels.at(1).copyTo(S_Mat);
	//channels.at(2).copyTo(V_Mat);
	//imshow("H", H_Mat);
	//imshow("S", S_Mat);
	//imshow("V", V_Mat);

	//分离HSV通道三个得出S通道最适合进行处理m，并且发现火焰一直处于右下角，故遍历像素只需右下角即可

	//遍历像素
	int row = srcMat.rows;            //行数
	int col = srcMat.cols;            //列数
	for (int i = row * 0.75; i < row; i++)    //行循环
	{
		for (int j = col * 0.75; j < col; j++)    //列循环
		{
			//开始处理每个像素
			if ((gray.at<uchar>(i, j) >= 150 && S_Mat.at<uchar>(i, j) >= 120))
			{
				dstMat.at<uchar>(i, j) = 255;
			}
			//处理结束
		}
	}
	//膨胀处理
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));    //定义内核矩阵，尺寸为15 15的矩形
	dilate(dstMat, dstMat, element);//膨胀

	//通过findContours函数寻找连通域
	vector<vector<Point>> contours;
	findContours(dstMat, contours, RETR_LIST, CHAIN_APPROX_NONE);

	//描绘轮廓
	for (int i = 0; i < contours.size(); i++) {
		drawContours(srcMat, contours, i, Scalar(0, 0, 255), 1, 8);

	}
}

//曲线拟合，因上课无曲线拟合内容，参考CSDN博客： https://blog.csdn.net/guduruyu/article/details/72866144
bool curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
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