#include<iostream>
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

//图片存储路径
const string path = "../temp/";
//保存路径
vector<string> imgs_path;
//帧间隔
int frameInterval = 400;

//
Mat result;


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
			imgs_path.emplace_back(path_buf);
			imwrite(path_buf, frame);
		}
		count++;
	}
	return true;
}

int main() {

	
	string videoPath = "../video/250飞行高度.MOV";
	//初始化数据
	init();
	//从视频中按帧间隔获取图片
	getImages(videoPath);

	vector<Mat> images;
	for (int i = 0; i < imgs_path.size();i++) {
		images.emplace_back(imread(imgs_path[i]));
	}
	

	int image_num = images.size();
	
	Ptr<FeaturesFinder> finder;
	//采用ORB算法寻找特征点
	finder = new OrbFeaturesFinder();
	vector<ImageFeatures> features(image_num);

	int64 tt = getTickCount();

	for (int i=0;i<images.size();i++)
	{
		Mat img = images[i];
 		(*finder)(img, features[i]);
 		features[i].img_idx = i;
 		cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;

	}
	//释放内存
	finder->collectGarbage();
	cout << "Finding features times:" << (getTickCount() - tt) / getTickFrequency() << "sec" << endl;
	 
	//特征点匹配
	vector<MatchesInfo> pair_matches;
	BestOf2NearestMatcher matcher(true,0.3f);
	matcher(features, pair_matches);
	matcher.collectGarbage();
	cout << pair_matches[0].H << endl;

	return 0;
}