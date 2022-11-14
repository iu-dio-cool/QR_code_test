// QR_code_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#if 0
#include <iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace cv;
using namespace std;



//获取轮廓的中心点
Point Center_cal(vector<vector<Point> > contours, int i)
{
    int centerx = 0, centery = 0, n = contours[i].size();
    //在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，
    //求所提取四个点的平均坐标（即为小正方形的大致中心）
    centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
    centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
    Point point1 = Point(centerx, centery);
    return point1;
}

int main()
{
    //std::cout << "Hello World!\n";

//特征识别
    Mat src = imread("D:\\ALL_aboutSWU\\cat_dogs\\456.jpg", 1);
    if (src.empty())
    {
        fprintf(stderr, "Can not load image！\n");
        return 0;
    }

    Mat src_all = src.clone();

    //彩色图转灰度图  
    Mat src_gray;
    cvtColor(src, src_gray, CV_BGR2GRAY);

    //对图像进行平滑处理  
    blur(src_gray, src_gray, Size(3, 3));

    //使灰度图象直方图均衡化  
    equalizeHist(src_gray, src_gray);

    namedWindow("src_gray");
    imshow("src_gray", src_gray);         //灰度图

    //指定112阀值进行二值化
    Mat threshold_output;
    threshold(src_gray, threshold_output, 112, 255, THRESH_BINARY);


    namedWindow("二值化后输出");
    imshow("二值化后输出", threshold_output);   //二值化后输出


    //需要的变量定义
    Scalar color = Scalar(1, 1, 255);
    vector<vector<Point>> contours, contours2;
    vector<Vec4i> hierarchy;
    //Mat drawing = Mat::zeros(src.size(), CV_8UC3);
    //Mat imageContours = Mat::ones(src_gray.size(), CV_8UC1); //最小外接矩形画布 
    Mat drawingAllContours = Mat::zeros(src_gray.size(), CV_8UC3);

    findContours(src_gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point());

    int flag = 0, c = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        if (hierarchy[i][2] != -1 && flag == 0)
        {
            flag++;
            c = i;

        }

        else if (hierarchy[i][2] == -1)
        {
            flag = 0;
        }

        else if (hierarchy[i][2] != -1)
        {
            flag++;
        }

        if (flag >= 2)
        {
            flag = 0;
            contours2.push_back(contours[c]);
        }
    }

    int count = contours2.size();
    cout << count << endl;
    vector<Point> pointthree;
    for (int i = 0; i < count; i++) {
        RotatedRect rect = minAreaRect(contours2[i]);

        Point2f P[4];
        rect.points(P);
        //		circle(imageContours, P[1], 6, (255), 1, 8);
        for (int j = 0; j <= 3; j++)
        {
            line(drawingAllContours, P[j], P[(j + 1) % 4], Scalar(255), 2);

        }
        imshow("MinAreaRect", drawingAllContours);

        pointthree.push_back(rect.center);
    }
#if 0
    //利用二值化输出寻找轮廓
    findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

    //寻找轮廓的方法
    int tempindex1 = 0;
    int tempindex2 = 0;
    vector<int> vin;
    int in[3];
    for (int i = 0; i < contours.size(); i++)
    {
        if (hierarchy[i][2] == -1)
            continue;
        else
            tempindex1 = hierarchy[i][2];                //第一个子轮廓的索引

        if (hierarchy[tempindex1][2] == -1)
            continue;
        else
        {
            tempindex2 = hierarchy[tempindex1][2];        //第二个子轮廓的索引
            //记录搜索到的有两个子轮廓的轮廓并把他们的编号存储
            in[0] = i;
            in[1] = tempindex1;
            in[2] = tempindex2;
            vin.push_back(in);
        }
    }
    
    //按面积比例搜索
    vector<index>::iterator it;
    for (it = vin.begin(); it != vin.end();)
    {
        vector<Point> out1Contours = contours[it->a1];
        vector<Point> out2Contours = contours[it->a2];
        double lenth1 = arcLength(out1Contours, 1);
        double lenth2 = arcLength(out2Contours, 1);
        if (abs(lenth1 / lenth2 - 2) > 1)
        {
            it = vin.erase(it);
        }
        else
        {
            drawContours(drawing, contours, it->a1, CV_RGB(255, 255, 255), 1, 8);
            it++;
        }
    }

    //获取三个定位角的中心坐标  
    Point point[3];
    int i = 0;
    vector<Point> pointthree;
    for (it = vin.begin(), i = 0; it != vin.end(); i++, it++)
    {
        point[i] = Center_cal(contours, it->a1);
        pointthree.push_back(point[i]);
    }

    if (pointthree.size() < 3)
    {
        cout << "找到的定位角点不足3个" << endl;
        return 0;
    }

    //计算轮廓的面积，计算定位角的面积，从而计算出边长
    double area = contourArea(contours[vin[0].a1]);
    int area_side = cvRound(sqrt(double(area)));
    for (int i = 0; i < 3; i++)
    {
        //画出三个定位角的中心连线  
        line(drawing, point[i % 3], point[(i + 1) % 3], color, area_side / 10, 8);
    }

    //清除找到的3个点,以便处理下一幅图片使用
    vin.clear();
#endif
    //由3个定位角校正图片
    //=========================================
    //找到角度最大的点
    double ca[2];
    double cb[2];

    ca[0] = pointthree[1].x - pointthree[0].x;
    ca[1] = pointthree[1].y - pointthree[0].y;
    cb[0] = pointthree[2].x - pointthree[0].x;
    cb[1] = pointthree[2].y - pointthree[0].y;
    double angle1 = 180 / 3.1415 * acos((ca[0] * cb[0] + ca[1] * cb[1]) / (sqrt(ca[0] * ca[0] + ca[1] * ca[1]) * sqrt(cb[0] * cb[0] + cb[1] * cb[1])));
    double ccw1;
    if (ca[0] * cb[1] - ca[1] * cb[0] > 0) ccw1 = 0;
    else ccw1 = 1;

    ca[0] = pointthree[0].x - pointthree[1].x;
    ca[1] = pointthree[0].y - pointthree[1].y;
    cb[0] = pointthree[2].x - pointthree[1].x;
    cb[1] = pointthree[2].y - pointthree[1].y;
    double angle2 = 180 / 3.1415 * acos((ca[0] * cb[0] + ca[1] * cb[1]) / (sqrt(ca[0] * ca[0] + ca[1] * ca[1]) * sqrt(cb[0] * cb[0] + cb[1] * cb[1])));
    double ccw2;
    if (ca[0] * cb[1] - ca[1] * cb[0] > 0) ccw2 = 0;
    else ccw2 = 1;

    ca[0] = pointthree[1].x - pointthree[2].x;
    ca[1] = pointthree[1].y - pointthree[2].y;
    cb[0] = pointthree[0].x - pointthree[2].x;
    cb[1] = pointthree[0].y - pointthree[2].y;
    double angle3 = 180 / 3.1415 * acos((ca[0] * cb[0] + ca[1] * cb[1]) / (sqrt(ca[0] * ca[0] + ca[1] * ca[1]) * sqrt(cb[0] * cb[0] + cb[1] * cb[1])));
    double ccw3;
    if (ca[0] * cb[1] - ca[1] * cb[0] > 0) ccw3 = 0;
    else ccw3 = 1;

    vector<Point2f> poly(4);
    if (angle3 > angle2 && angle3 > angle1)
    {
        if (ccw3)
        {
            poly[1] = pointthree[1];
            poly[3] = pointthree[0];
        }
        else
        {
            poly[1] = pointthree[0];
            poly[3] = pointthree[1];
        }
        poly[0] = pointthree[2];
        Point temp(pointthree[0].x + pointthree[1].x - pointthree[2].x, pointthree[0].y + pointthree[1].y - pointthree[2].y);
        poly[2] = temp;
    }
    else if (angle2 > angle1 && angle2 > angle3)
    {
        if (ccw2)
        {
            poly[1] = pointthree[0];
            poly[3] = pointthree[2];
        }
        else
        {
            poly[1] = pointthree[2];
            poly[3] = pointthree[0];
        }
        poly[0] = pointthree[1];
        Point temp(pointthree[0].x + pointthree[2].x - pointthree[1].x, pointthree[0].y + pointthree[2].y - pointthree[1].y);
        poly[2] = temp;
    }
    else if (angle1 > angle2 && angle1 > angle3)
    {
        if (ccw1)
        {
            poly[1] = pointthree[1];
            poly[3] = pointthree[2];
        }
        else
        {
            poly[1] = pointthree[2];
            poly[3] = pointthree[1];
        }
        poly[0] = pointthree[0];
        Point temp(pointthree[1].x + pointthree[2].x - pointthree[0].x, pointthree[1].y + pointthree[2].y - pointthree[0].y);
        poly[2] = temp;
    }

    vector<Point2f> trans(4);
    int temp = 50;
    trans[0] = Point2f(0 + temp, 0 + temp);
    trans[1] = Point2f(0 + temp, 100 + temp);
    trans[2] = Point2f(100 + temp, 100 + temp);
    trans[3] = Point2f(100 + temp, 0 + temp);

    //获取透视投影变换矩阵
    //CvMat* warp_mat = cvCreateMat(3, 3, CV_32FC1);
    //cvGetPerspectiveTransform(poly, trans, warp_mat);
    Mat m = getPerspectiveTransform(poly, trans);
    //计算变换结果
    //IplImage ipl_img(src_all);
    //IplImage* dst = cvCreateImage(cvSize(1000, 1000), 8, 3);
    //cvWarpPerspective(&ipl_img, dst, warp_mat);
    Mat result;
    warpPerspective(src_all, result, m, Size(350, 350), INTER_LINEAR);   //**********

    rectangle(result, Rect(10, 10, 330, 330), Scalar(0, 0, 0), 1, 8);
    //=========================================

#if 1
    string pathtemp = "32131";
    namedWindow("透视变换后的图");
    cvShowImage("透视变换后的图", dst);         //透视变换后的图

    drawContours(drawingAllContours, contours, -1, CV_RGB(255, 255, 255), 1, 8);
    namedWindow("DrawingAllContours");
    imshow("DrawingAllContours", drawingAllContours);

    namedWindow(pathtemp);
    imshow(pathtemp, drawing);    //3个角点填充
#endif

    //接下来要框出这整个二维码  
    Mat gray_all, threshold_output_all;
    vector<vector<Point> > contours_all;
    vector<Vec4i> hierarchy_all;
    //cvtColor(drawing, gray_all, CV_BGR2GRAY);

    threshold(gray_all, threshold_output_all, 45, 255, THRESH_BINARY);

    findContours(threshold_output_all, contours_all, hierarchy_all, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));//RETR_EXTERNAL表示只寻找最外层轮廓  

    Point2f fourPoint2f[4];
    //求最小包围矩形  
    RotatedRect rectPoint = minAreaRect(contours_all[0]);

    //将rectPoint变量中存储的坐标值放到 fourPoint的数组中  
    rectPoint.points(fourPoint2f);
    for (int i = 0; i < 4; i++)
    {
        line(src_all, fourPoint2f[i % 4], fourPoint2f[(i + 1) % 4],
            Scalar(20, 21, 237), 3);
    }
    
    namedWindow(pathtemp);
    imshow(pathtemp, src_all);

    //截取二维码区域
    CvSize size = cvSize(200, 200);//区域大小
    //cvSetImageROI(dst, cvRect(0, 0, size.width, size.height));//设置源图像ROI
    //IplImage* pDest = cvCreateImage(size, dst->depth, dst->nChannels);//创建目标图像
    
    //cvCopy(dst, pDest); //复制图像
    // cvSaveImage("Roi.jpg", pDest);//保存目标图像

#if 0
 //信息识别
        //对截取后的区域进行解码
    Mat imageSource = cv::Mat(pDest);
    cvResetImageROI(pDest);//源图像用完后，清空ROI
    cvtColor(imageSource, imageSource, CV_BGR2GRAY);  //zbar需要输入灰度图像才能很好的识别

    //Zbar二维码识别
    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
    int width1 = imageSource.cols;
    int height1 = imageSource.rows;
    uchar* raw = (uchar*)imageSource.data;

    Image imageZbar(width1, height1, "Y800", raw, width1* height1);
    scanner.scan(imageZbar); //扫描条码      
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        cout << "查询条码失败，请检查图片！" << endl;
    }

    for (; symbol != imageZbar.symbol_end(); ++symbol)
    {
        cout << "类型：" << endl << symbol->get_type_name() << endl;
        cout << "条码：" << endl << symbol->get_data() << endl;
    }

    imageZbar.set_data(NULL, 0);

#endif


}


#endif

#if 1
#include <opencv2/opencv.hpp>
#include <iostream>    
#include <opencv2\core\core.hpp>
#include <stdio.h>
#include <string>
#include <sstream>
#include <zbar.h>
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;


// 用于矫正
Mat transformCorner(Mat src, RotatedRect rect);
Mat transformQRcode(Mat src, RotatedRect rect, double angle);
// 用于判断角点
bool IsQrPoint(vector<Point>& contour, Mat& img);
bool isCorner(Mat& image);
double Rate(Mat& count);
int leftTopPoint(vector<Point> centerPoint);
vector<int> otherTwoPoint(vector<Point> centerPoint, int leftTopPointIndex);
double rotateAngle(Point leftTopPoint, Point rightTopPoint, Point leftBottomPoint);
//otherTwoPointIndex返回二维码对应的意义

int main()
{
	//VideoCapture cap;
	//Mat src;
	//cap.open(0);                             //打开相机，电脑自带摄像头一般编号为0，外接摄像头编号为1，主要是在设备管理器中查看自己摄像头的编号。

	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);  //设置捕获视频的宽度
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 400);  //设置捕获视频的高度

	//if (!cap.isOpened())                         //判断是否成功打开相机
	//{
	//	cout << "摄像头打开失败!" << endl;
		//return -1;
	//}
	while (1) {
		Mat src;
		src = imread("D:\\ALL_aboutSWU\\cat_dogs\\456.jpg");  //打开图片
		//cap >> src;                                //从相机捕获一帧图像


		Mat srcCopy = src.clone();

		//canvas为画布 将找到的定位特征画出来
		Mat canvas;
		canvas = Mat::zeros(src.size(), CV_8UC3);

		Mat srcGray;

		//center_all获取特性中心
		vector<Point> center_all;

		// 转化为灰度图
		cvtColor(src, srcGray, COLOR_BGR2GRAY);

		// 3X3模糊
		blur(srcGray, srcGray, Size(3, 3));

		// 计算直方图
		convertScaleAbs(src, src);
		equalizeHist(srcGray, srcGray);
		int s = srcGray.at<Vec3b>(0, 0)[0];
		// 设置阈值根据实际情况 如视图中已找不到特征 可适量调整
		threshold(srcGray, srcGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
		imshow("threshold", srcGray);

		/*contours是第一次寻找轮廓*/
		/*contours2是筛选出的轮廓*/
		vector<vector<Point>> contours;

		//	用于轮廓检测
		vector<Vec4i> hierarchy;
		findContours(srcGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		// 小方块的数量
		int numOfRec = 0;

		// 检测方块
		int ic = 0;
		int parentIdx = -1;
		for (int i = 0; i < contours.size(); i++)
		{
			if (hierarchy[i][2] != -1 && ic == 0)
			{
				parentIdx = i;
				ic++;
			}
			else if (hierarchy[i][2] != -1)
			{
				ic++;
			}
			else if (hierarchy[i][2] == -1)
			{
				parentIdx = -1;
				ic = 0;
			}
			if (ic >= 2 && ic <= 2)
			{
				if (IsQrPoint(contours[parentIdx], src)) {
					RotatedRect rect = minAreaRect(Mat(contours[parentIdx]));

					// 画图部分
					Point2f points[4];
					rect.points(points);
					for (int j = 0; j < 4; j++) {
						line(src, points[j], points[(j + 1) % 4], Scalar(0, 255, 0), 2);
					}
					drawContours(canvas, contours, parentIdx, Scalar(0, 0, 255), -1);

					// 如果满足条件则存入
					center_all.push_back(rect.center);
					numOfRec++;
				}
				ic = 0;
				parentIdx = -1;
			}
		}

		// 连接三个正方形的部分
		for (int i = 0; i < center_all.size(); i++)
		{
			line(canvas, center_all[i], center_all[(i + 1) % center_all.size()], Scalar(255, 0, 0), 3);
		}

		vector<vector<Point>> contours3;
		Mat canvasGray;
		cvtColor(canvas, canvasGray, COLOR_BGR2GRAY);
		findContours(canvasGray, contours3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		vector<Point> maxContours;
		double maxArea = 0;
		// 在原图中画出二维码的区域

		for (int i = 0; i < contours3.size(); i++)
		{
			RotatedRect rect = minAreaRect(contours3[i]);
			Point2f boxpoint[4];
			rect.points(boxpoint);
			for (int i = 0; i < 4; i++)
				line(src, boxpoint[i], boxpoint[(i + 1) % 4], Scalar(0, 0, 255), 3);

			double area = contourArea(contours3[i]);
			if (area > maxArea) {
				maxContours = contours3[i];
				maxArea = area;
			}
		}
		imshow("src", src);
		if (numOfRec < 3) {
			waitKey(10);
			continue;
		}
		// 计算“回”的次序关系
		int leftTopPointIndex = leftTopPoint(center_all);
		vector<int> otherTwoPointIndex = otherTwoPoint(center_all, leftTopPointIndex);
		// canvas上标注三个“回”的次序关系
		circle(canvas, center_all[leftTopPointIndex], 10, Scalar(255, 0, 255), -1);
		circle(canvas, center_all[otherTwoPointIndex[0]], 10, Scalar(0, 255, 0), -1);
		circle(canvas, center_all[otherTwoPointIndex[1]], 10, Scalar(0, 255, 255), -1);

		// 计算旋转角
		double angle = rotateAngle(center_all[leftTopPointIndex], center_all[otherTwoPointIndex[0]], center_all[otherTwoPointIndex[1]]);

		// 拿出之前得到的最大的轮廓
		RotatedRect rect = minAreaRect(Mat(maxContours));
		Mat image = transformQRcode(srcCopy, rect, angle);

		// 展示图像
		imshow("QRcode", image);
		imshow("canvas", canvas);
		waitKey(10);

		//对截取后的区域进行解码
		Mat imageSource = cv::Mat(image);
		//cvResetImageROI(image);//源图像用完后，清空ROI
		cvtColor(imageSource, imageSource, CV_BGR2GRAY);  //zbar需要输入灰度图像才能很好的识别

		//Zbar二维码识别
		zbar::ImageScanner scanner;
		scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
		int width1 = imageSource.cols;
		int height1 = imageSource.rows;
		uchar* raw = (uchar*)imageSource.data;

		zbar::Image imageZbar(width1, height1, "Y800", raw, width1 * height1);
		scanner.scan(imageZbar); //扫描条码      
		zbar::Image::SymbolIterator symbol = imageZbar.symbol_begin();
		if (imageZbar.symbol_begin() == imageZbar.symbol_end())
		{
			cout << "查询条码失败，请检查图片！" << endl;
		}

		for (; symbol != imageZbar.symbol_end(); ++symbol)
		{
			cout << "类型：" << endl << symbol->get_type_name() << endl;
			cout << "条码：" << endl << symbol->get_data() << endl;
		}

		imageZbar.set_data(NULL, 0);


	}
	return 0;
}

Mat transformCorner(Mat src, RotatedRect rect)
{
	// 获得旋转中心
	Point center = rect.center;
	// 获得左上角和右下角的角点，而且要保证不超出图片范围，用于抠图
	Point TopLeft = Point(cvRound(center.x), cvRound(center.y)) - Point(rect.size.height / 2, rect.size.width / 2);  //旋转后的目标位置
	TopLeft.x = TopLeft.x > src.cols ? src.cols : TopLeft.x;
	TopLeft.x = TopLeft.x < 0 ? 0 : TopLeft.x;
	TopLeft.y = TopLeft.y > src.rows ? src.rows : TopLeft.y;
	TopLeft.y = TopLeft.y < 0 ? 0 : TopLeft.y;

	int after_width, after_height;
	if (TopLeft.x + rect.size.width > src.cols) {
		after_width = src.cols - TopLeft.x - 1;
	}
	else {
		after_width = rect.size.width - 1;
	}
	if (TopLeft.y + rect.size.height > src.rows) {
		after_height = src.rows - TopLeft.y - 1;
	}
	else {
		after_height = rect.size.height - 1;
	}
	// 获得二维码的位置
	Rect RoiRect = Rect(TopLeft.x, TopLeft.y, after_width, after_height);

	//	dst是被旋转的图片 roi为输出图片 mask为掩模
	double angle = rect.angle;
	Mat mask, roi, dst;
	Mat image;
	// 建立中介图像辅助处理图像

	vector<Point> contour;
	// 获得矩形的四个点
	Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++)
		contour.push_back(points[i]);

	vector<vector<Point>> contours;
	contours.push_back(contour);
	// 再中介图像中画出轮廓
	drawContours(mask, contours, 0, Scalar(255, 255, 255), -1);
	// 通过mask掩膜将src中特定位置的像素拷贝到dst中。
	src.copyTo(dst, mask);
	// 旋转
	Mat M = getRotationMatrix2D(center, angle, 1);
	warpAffine(dst, image, M, src.size());
	// 截图
	roi = image(RoiRect);

	return roi;
}

// 该部分用于检测是否是角点，与下面两个函数配合
bool IsQrPoint(vector<Point>& contour, Mat& img) {
	double area = contourArea(contour);
	// 角点不可以太小
	if (area < 30)
		return 0;
	RotatedRect rect = minAreaRect(Mat(contour));
	double w = rect.size.width;
	double h = rect.size.height;
	double rate = min(w, h) / max(w, h);
	if (rate > 0.7)
	{
		// 返回旋转后的图片，用于把“回”摆正，便于处理
		Mat image = transformCorner(img, rect);
		if (isCorner(image))
		{
			return 1;
		}
	}
	return 0;
}

// 计算内部所有白色部分占全部的比率
double Rate(Mat& count)
{
	int number = 0;
	int allpixel = 0;
	for (int row = 0; row < count.rows; row++)
	{
		for (int col = 0; col < count.cols; col++)
		{
			if (count.at<uchar>(row, col) == 255)
			{
				number++;
			}
			allpixel++;
		}
	}
	//cout << (double)number / allpixel << endl;
	return (double)number / allpixel;
}

// 用于判断是否属于角上的正方形
bool isCorner(Mat& image)
{
	// 定义mask
	Mat imgCopy, dstCopy;
	Mat dstGray;
	imgCopy = image.clone();
	// 转化为灰度图像
	cvtColor(image, dstGray, COLOR_BGR2GRAY);
	// 进行二值化

	threshold(dstGray, dstGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
	dstCopy = dstGray.clone();  //备份

	// 找到轮廓与传递关系
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dstCopy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);


	for (int i = 0; i < contours.size(); i++)
	{
		//cout << i << endl;
		if (hierarchy[i][2] == -1 && hierarchy[i][3])
		{
			Rect rect = boundingRect(Mat(contours[i]));
			rectangle(image, rect, Scalar(0, 0, 255), 2);
			// 最里面的矩形与最外面的矩形的对比
			if (rect.width < imgCopy.cols * 2 / 7)      //2/7是为了防止一些微小的仿射
				continue;
			if (rect.height < imgCopy.rows * 2 / 7)      //2/7是为了防止一些微小的仿射
				continue;
			// 判断其中黑色与白色的部分的比例
			if (Rate(dstGray) > 0.20)
			{
				return true;
			}
		}
	}
	return  false;
}

int leftTopPoint(vector<Point> centerPoint) {
	int minIndex = 0;
	int multiple = 0;
	int minMultiple = 10000;
	multiple = (centerPoint[1].x - centerPoint[0].x) * (centerPoint[2].x - centerPoint[0].x) + (centerPoint[1].y - centerPoint[0].y) * (centerPoint[2].y - centerPoint[0].y);
	if (minMultiple > multiple) {
		minIndex = 0;
		minMultiple = multiple;
	}
	multiple = (centerPoint[0].x - centerPoint[1].x) * (centerPoint[2].x - centerPoint[1].x) + (centerPoint[0].y - centerPoint[1].y) * (centerPoint[2].y - centerPoint[1].y);
	if (minMultiple > multiple) {
		minIndex = 1;
		minMultiple = multiple;
	}
	multiple = (centerPoint[0].x - centerPoint[2].x) * (centerPoint[1].x - centerPoint[2].x) + (centerPoint[0].y - centerPoint[2].y) * (centerPoint[1].y - centerPoint[2].y);
	if (minMultiple > multiple) {
		minIndex = 2;
		minMultiple = multiple;
	}
	return minIndex;
}

vector<int> otherTwoPoint(vector<Point> centerPoint, int leftTopPointIndex) {
	vector<int> otherIndex;
	double waiji = (centerPoint[(leftTopPointIndex + 1) % 3].x - centerPoint[(leftTopPointIndex) % 3].x) *
		(centerPoint[(leftTopPointIndex + 2) % 3].y - centerPoint[(leftTopPointIndex) % 3].y) -
		(centerPoint[(leftTopPointIndex + 2) % 3].x - centerPoint[(leftTopPointIndex) % 3].x) *
		(centerPoint[(leftTopPointIndex + 1) % 3].y - centerPoint[(leftTopPointIndex) % 3].y);
	if (waiji > 0) {
		otherIndex.push_back((leftTopPointIndex + 1) % 3);
		otherIndex.push_back((leftTopPointIndex + 2) % 3);
	}
	else {
		otherIndex.push_back((leftTopPointIndex + 2) % 3);
		otherIndex.push_back((leftTopPointIndex + 1) % 3);
	}
	return otherIndex;
}

double rotateAngle(Point leftTopPoint, Point rightTopPoint, Point leftBottomPoint) {
	double dy = rightTopPoint.y - leftTopPoint.y;
	double dx = rightTopPoint.x - leftTopPoint.x;
	double k = dy / dx;
	double angle = atan(k) * 180 / CV_PI;//转化角度
	if (leftBottomPoint.y < leftTopPoint.y)
		angle -= 180;
	return angle;
}

Mat transformQRcode(Mat src, RotatedRect rect, double angle)
{
	// 获得旋转中心
	Point center = rect.center;
	// 获得左上角和右下角的角点，而且要保证不超出图片范围，用于抠图
	Point TopLeft = Point(cvRound(center.x), cvRound(center.y)) - Point(rect.size.height / 2, rect.size.width / 2);  //旋转后的目标位置
	TopLeft.x = TopLeft.x > src.cols ? src.cols : TopLeft.x;
	TopLeft.x = TopLeft.x < 0 ? 0 : TopLeft.x;
	TopLeft.y = TopLeft.y > src.rows ? src.rows : TopLeft.y;
	TopLeft.y = TopLeft.y < 0 ? 0 : TopLeft.y;

	int after_width, after_height;
	if (TopLeft.x + rect.size.width > src.cols) {
		after_width = src.cols - TopLeft.x - 1;
	}
	else {
		after_width = rect.size.width - 1;
	}
	if (TopLeft.y + rect.size.height > src.rows) {
		after_height = src.rows - TopLeft.y - 1;
	}
	else {
		after_height = rect.size.height - 1;
	}
	// 获得二维码的位置
	Rect RoiRect = Rect(TopLeft.x, TopLeft.y, after_width, after_height);

	// dst是被旋转的图片，roi为输出图片，mask为掩模
	Mat mask, roi, dst;
	Mat image;
	// 建立中介图像辅助处理图像

	vector<Point> contour;
	// 获得矩形的四个点
	Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++)
		contour.push_back(points[i]);

	vector<vector<Point>> contours;
	contours.push_back(contour);
	// 再中介图像中画出轮廓
	drawContours(mask, contours, 0, Scalar(255, 255, 255), -1);
	// 通过mask掩膜将src中特定位置的像素拷贝到dst中。
	src.copyTo(dst, mask);
	// 旋转
	Mat M = getRotationMatrix2D(center, angle, 1);
	warpAffine(dst, image, M, src.size());
	// 截图
	roi = image(RoiRect);

	return roi;
}


#endif


#if 0
#include <opencv2/opencv.hpp>
#include <iostream>    
#include <opencv2\core\core.hpp>
#include <stdio.h>
#include <string>
#include <sstream>
#include <zbar.h>
#include <opencv2\imgproc\types_c.h>
#include <D:\ALL_aboutSWU\cat_dog_lab\QR_code\qrcode1\QR_code_test\iconv.h>

#pragma comment(lib,"D:\\ALL_aboutSWU\\cat_dog_lab\\QR_code\\qrcode1\\QR_code_test\\libiconv.lib")
using namespace zbar;
using namespace cv;
using namespace std;

int main()
{
	clock_t start = clock(); // 记录程序开始时间，用于计算扫描二维码耗时
	zbar::ImageScanner scanner;
	scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
	Mat imgOrigin = imread("D:\\ALL_aboutSWU\\cat_dogs\\789.jpg");  // 二维码图片相对路径
	Mat imgGray;
	cvtColor(imgOrigin, imgGray, CV_BGR2GRAY);  // 灰度化


	int width = imgGray.cols;
	int height = imgGray.rows;
	Image image(width, height, "Y800", imgGray.data, width * height);  // 图片格式转换
	scanner.scan(image);
	Image::SymbolIterator symbol = image.symbol_begin();
	if (image.symbol_begin() == image.symbol_end())
	{
		cout << "查询条码失败，请检查图片！" << endl;
	}
	for (; symbol != image.symbol_end(); ++symbol)
	{
		cout << "类型：" << endl << symbol->get_type_name() << endl << endl;
		cout << "条码：" << endl << symbol->get_data() << endl << endl;

		cout << "x:" << symbol->get_location_x(0) << endl << endl;
		cout << "y:" << symbol->get_location_y(0) << endl << endl;
		//定位信息，
		int x = symbol->get_location_x(0);
		int y = symbol->get_location_y(0);

		//根据zbar 返回的多边形的像素点位置 计算宽高 
		//默认二维码是垂直水平的
		int min_x = 0, min_y = 0, max_x = 0, max_y = 0;
		Symbol::PointIterator pt = symbol->point_begin();
		Symbol::Point p = *pt;

		min_x = p.x;
		min_y = p.y;
		max_x = p.x;
		max_y = p.y;

		for (; pt != (Symbol::PointIterator)symbol->point_end(); ++pt) {
			p = *pt;
			min_x = min_x < p.x ? min_x : p.x;
			max_x = min_x > p.x ? max_x : p.x;

			min_y = min_y < p.y ? min_y : p.y;
			max_y = max_y > p.y ? max_y : p.y;
		}


		cout << "width:" << max_x - min_x << endl << endl;
		cout << "height:" << max_y - min_y << endl << endl;

		cv::Rect r(x, y, max_x - min_x, max_y - min_y);
		cv::rectangle(imgOrigin, r, Scalar(255, 0, 0), 8, LINE_8, 0);

	}
	image.set_data(nullptr, 0);

	clock_t finish = clock();  // 记录程序结束时间
	double time_length = (double)(finish - start) / CLOCKS_PER_SEC; //根据两个时刻的差，计算出扫描的时间  
	cout << "扫描耗时 " << time_length << " seconds." << endl;


	imshow("原图", imgOrigin);

	waitKey();
	return 0;
}

#endif






// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
