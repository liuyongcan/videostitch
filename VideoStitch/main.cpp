#include<iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <QDir>
#include <QString>

using namespace std;
using namespace cv;
using namespace detail;

//视频目录
string videoPath = "../video/250飞行高度.MOV";
//图片存储路径
const string path = "../temp/";
//保存路径
vector<String> imgs_path;
//选取帧间隔特征点匹配下限----与飞行速度有关（速度越快，下限越大，帧间隔越小）
int matchKeyPointNum_Min = 450;
//帧间隔
int frameInterval;

//初始化，准备工作目录
void init() {

	QDir dir(QString::fromStdString(path));
	if (dir.exists())
	{
		//清空文件内容
		dir.setFilter(QDir::Files);
		for (int i = 0; i < dir.count(); i++)
			dir.remove(dir[i]);
	}
	else {
		//创建存储文件夹
		dir.mkdir(QString::fromStdString(path));
	}

}

//获取两张图片的特征点匹配数
int findMatchPoints(Mat src_1,Mat src_2) {

	Ptr<FeaturesFinder> finder;
	//采用ORB算法寻找特征点
	finder = new OrbFeaturesFinder();
	vector<ImageFeatures> features(2);
	
	Mat grayP1, grayP2;
	vector<Mat> images_gray;
	//灰度化
	cvtColor(src_1, grayP1, CV_BGR2GRAY);
	images_gray.emplace_back(grayP1);
	cvtColor(src_2, grayP2, CV_BGR2GRAY);
	images_gray.emplace_back(grayP2);
	
	//寻找特征点
	for (int i = 0; i < images_gray.size(); i++)
	{
		Mat img = images_gray[i];
		(*finder)(img, features[i]);
		features[i].img_idx = i;
		cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
	}
	//释放内存
	finder->collectGarbage();
	//特征点匹配
	vector<MatchesInfo> pair_matches;
	BestOf2NearestMatcher matcher(false);
	matcher(features, pair_matches);
	matcher.collectGarbage();
	cout << "匹配数：" << pair_matches[1].num_inliers << endl;
	return pair_matches[1].num_inliers;
}

//自动获取合适的帧间隔
void getFrameInterval() {

	Mat frame;
	size_t count = 0;
	
	cv::VideoCapture capture(videoPath);
	if (!capture.isOpened())
	{
		cout << "视频文件无法打开！！";
		return ;
	}
	Mat src_1,src_2;

	while (capture.read(frame))
	{
		
		if (frame.empty())
			break;

		if (src_1.empty())
		{
			frame.copyTo(src_1);
			
		}
		else
		{
			if (count % 50 == 0) {
				frame.copyTo(src_2);
				
				int num = findMatchPoints(src_1, src_2);
				if (num <= matchKeyPointNum_Min)
				{
					frameInterval = count;
					cout << "合适的帧间隔：" << count << endl;
					return;
				}
			}
		}
		count++;
	}


}

//根据视频获取图像
bool getImages(string videoPath) {

	Mat frame;
	size_t count = 0;
	stringstream ss;
	string path_buf;

	cv::VideoCapture capture(videoPath);
	if (!capture.isOpened())
	{
		cout << "视频文件无法打开！！";
		return false;
	}

	imgs_path.clear();
	while (capture.read(frame))
	{
		if (frame.empty())
			break;

		if (count%frameInterval == 0)
		{
			ss.clear();
			ss << path << count << ".jpg";
			ss >> path_buf;
			cout << path_buf<<endl;
			imgs_path.emplace_back(path_buf);
			imwrite(path_buf, frame);
		}
		count++;
	}
	return true;
}

//图像拼接，将src_1拼接到src_2上
Mat stitchImage(Mat src_1, Mat src_2) {

	for (int i = 0; i < src_1.rows; i++) {
		uchar* row = src_1.ptr(i);
		uchar* row_dst = src_2.ptr(i);
		for (int j = 0; j < src_1.cols; j++) {

			if (row_dst[j * 3] == 0 && row_dst[j * 3 + 1] == 0 && row_dst[j * 3 + 2] == 0) {
				row_dst[j * 3] = row[j * 3];
				row_dst[j * 3 + 1] = row[j * 3 + 1];
				row_dst[j * 3 + 2] = row[j * 3 + 2];

			}
			else
			{
				row_dst[j * 3] = row_dst[j * 3];
				row_dst[j * 3 + 1] = row_dst[j * 3 + 1];
				row_dst[j * 3 + 2] = row_dst[j * 3 + 2];
			}
		}
	}
	return src_2;
}

int main() {

	//findMatchPoints(imread("D:/VS2015WorkSpace/OpenCV/VideoStitch/temp/600.jpg"), imread("D:/VS2015WorkSpace/OpenCV/VideoStitch/temp/3000.jpg"));

	cout << "-------开始读取视频数据-----------"<<endl<<endl;
	int64 tt = getTickCount();
	//初始化数据
	init();
	//自动获取合适的帧间隔
	getFrameInterval();
	//从视频中按帧间隔获取图片
	getImages(videoPath);
	cout << "\n-------结束读取视频数据-----------" << endl;
	cout << "get pictures times:" << (getTickCount() - tt) / getTickFrequency() << " sec" << endl;
	//原图像
	vector<Mat> images;
	//调整图像大小
	vector<Mat> images_resize;
	//灰度图像
	vector<Mat> images_gray;
	//拼接图像的大小
	const int images_num = imgs_path.size();

	tt = getTickCount();
	//调整图片位置
#pragma omp parallel for
	for (int i = 0; i < imgs_path.size(); i++)
	{
		images.emplace_back(imread(imgs_path[i]));

		Mat src_p = images[i];
		Mat imag(2500, 2500, CV_8UC3, Scalar::all(0));
		Mat temp(imag, Rect(800, 800, src_p.cols, src_p.rows));
		src_p.copyTo(temp);
		images_resize.emplace_back(imag);
		cvtColor(imag, imag, CV_BGR2GRAY);
		images_gray.emplace_back(imag);

	}
	cout << "init times:" << (getTickCount() - tt) / getTickFrequency() << " sec" << endl;

	tt = getTickCount();
	//最终图像
	Mat dst;
	//取第一张图像
	Mat src = images_resize[0];
	//ORB
	Ptr<ORB> orb = ORB::create();
#pragma omp parallel for
	for (int i = 1; i < images_num; i++)
	{
		//图像的特征点
		std::vector<KeyPoint> keypoints_src, keypoints_;
		//创建两张图像的描述子，类型是Mat类型
		Mat descriptors_src, descriptors_;
		
		//第一步：检测Oriented FAST角点位置.
		orb->detect(src, keypoints_src);
		orb->detect(images_resize[i], keypoints_);
		
		//第二步：根据角点位置计算BRIEF描述子
		orb->compute(src, keypoints_src, descriptors_src);
		orb->compute(images_resize[i], keypoints_, descriptors_);

		//匹配信息数组
		vector<DMatch> matches;
		BFMatcher matcher(NORM_HAMMING);
		//特征点匹配
		matcher.match(descriptors_src, descriptors_, matches);
		//获取匹配点
		vector<Point2f> imagePoints1, imagePoints2;
		for (int j = 0; j < matches.size(); j++)
		{
			imagePoints1.push_back(keypoints_src[matches[j].queryIdx].pt);
			imagePoints2.push_back(keypoints_[matches[j].trainIdx].pt);

		}

		//根据匹配点得到两张图像的变换矩阵
		Mat H;
		H = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
		//对图像进行透视变换, 就是变形
		Mat outImg;
		warpPerspective(src, outImg, H, Size(2000, 2000));
		//拼接
		dst=stitchImage(outImg, images_resize[i]);
		
		//清除数据
		keypoints_src.clear();
		keypoints_.clear();
		imagePoints1.clear();
		imagePoints2.clear();
		src.setTo(0);
		//将这次拼接结果作为下次拼接的起始条件
		dst.copyTo(src);
	}
	cout << "stitch picture times:" << (getTickCount() - tt) / getTickFrequency() << " sec" << endl;
	//矩阵转置
	//transpose(dst, dst);
	//旋转
	//flip(dst,dst,1);
	
	imwrite("res.jpg", dst);
	system("res.jpg");

}