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
#include <QtCore/QDir>

using namespace std;
using namespace cv;
using namespace detail;



//将src_1拼接到src_2上
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

	//保存路径
	vector<String> imgs_path;
	imgs_path.emplace_back("../temp/0.jpg");
	imgs_path.emplace_back("../temp/200.jpg");
	imgs_path.emplace_back("../temp/400.jpg");
	imgs_path.emplace_back("../temp/600.jpg");
	imgs_path.emplace_back("../temp/800.jpg");
	imgs_path.emplace_back("../temp/1000.jpg");
	imgs_path.emplace_back("../temp/1200.jpg");
	imgs_path.emplace_back("../temp/1400.jpg");
	imgs_path.emplace_back("../temp/1600.jpg");
	imgs_path.emplace_back("../temp/1800.jpg");
	imgs_path.emplace_back("../temp/2000.jpg");
	imgs_path.emplace_back("../temp/2200.jpg");
	imgs_path.emplace_back("../temp/2400.jpg");
	imgs_path.emplace_back("../temp/2600.jpg");
	imgs_path.emplace_back("../temp/2800.jpg");
	imgs_path.emplace_back("../temp/3000.jpg");
	imgs_path.emplace_back("../temp/3200.jpg");

	//原图像
	vector<Mat> images;
	//调整图像大小
	vector<Mat> images_resize;
	//灰度图像
	vector<Mat> images_gray;
	//
	const int images_num = imgs_path.size();

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

	Mat dst;
	Mat src = images_resize[0];

	
	//
	Ptr<ORB> orb = ORB::create();
	//
#pragma omp parallel for
	for (int i = 1; i < images_num; i++)
	{
		std::vector<KeyPoint> keypoints_src, keypoints_;
		//创建两张图像的描述子，类型是Mat类型
		Mat descriptors_src, descriptors_;
		
		Mat d,d2;
		//第一步：检测Oriented FAST角点位置.
		orb->detect(src, keypoints_src);
		drawKeypoints(src, keypoints_src, d, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		orb->detect(images_resize[i], keypoints_);
		drawKeypoints(images_resize[i], keypoints_, d2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//第二步：根据角点位置计算BRIEF描述子
		orb->compute(src, keypoints_src, descriptors_src);
		orb->compute(images_resize[i], keypoints_, descriptors_);



		vector<DMatch> matches;
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(descriptors_src, descriptors_, matches);

		vector<Point2f> imagePoints1, imagePoints2;
		for (int j = 0; j < matches.size(); j++)
		{
			imagePoints1.push_back(keypoints_src[matches[j].queryIdx].pt);
			imagePoints2.push_back(keypoints_[matches[j].trainIdx].pt);

		}

		Mat H;
		H = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
		Mat outImg;
		warpPerspective(src, outImg, H, Size(2000, 2000));
		dst=stitchImage(outImg, images_resize[i]);
		
		keypoints_src.clear();
		keypoints_.clear();
		imagePoints1.clear();
		imagePoints2.clear();
		src.setTo(0);
		dst.copyTo(src);

	}

	imwrite("res.jpg", dst);
	system("res.jpg");

}