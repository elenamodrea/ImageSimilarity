// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <cmath>
#include <opencv2/core/utils/logger.hpp>
#define M_PI 3.14159265358979323846
using namespace std;
wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
double gaussian(double x, double sigma) {
	return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI) * sigma);
}
const int K = 11;
void createGaussianFilter( double filter[K][K]) {

	int center = K / 2;

	double sum = 0.0;
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < K; j++) {
			double x = i - center;
			double y = j - center;
			filter[i][j] = gaussian(x, 1.5) * gaussian(y, 1.5);
			sum += filter[i][j];
		}
	}


	for (int i = 0; i < K; i++) {
		for (int j = 0; j < K; j++) {
			filter[i][j] /= sum;
		}
	}
}
Mat gaussianBlur(Mat src,int k) {
	int height = src.rows;
	int width = src.cols;
	
	double nucleu[K][K];
	createGaussianFilter(nucleu);
	Mat dst = Mat(src.rows, src.cols, CV_32F);
	
	for (int i = k/2; i < height - k/2; i++) {
		for (int j = k/2; j < width - k/2; j++) {
			float s = 0.0;
			for (int k1 = 0; k1 <k ; k1++) {
				for (int p = 0; p < k; p++) {
					s += src.at<float>(i + k1 - k/2, j + p - k/2) * nucleu[k1][p];
				}
			}
			dst.at<float>(i, j) = s;

		}
	}

	return dst;

}

void gaussianFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = gaussianBlur(src,11);
		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}
}

double ssimCalc(Mat img1, Mat img2) {
	const double C1 = 6.5025, C2 = 58.5225;
	const int H = img1.rows, W = img1.cols;

	Mat img1f, img2f;
	img1.convertTo(img1f, CV_32F);
	img2.convertTo(img2f, CV_32F);

	Mat img1f2 = img1f.mul(img1f), img2f2 = img2f.mul(img2f), img1f_img2f = img1f.mul(img2f);

	Mat mu1, mu2;
	mu1=gaussianBlur(img1f,11);
	mu2=gaussianBlur(img2f,11);

	Mat mu1_2 = mu1.mul(mu1), mu2_2 = mu2.mul(mu2), mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;
	sigma1_2=gaussianBlur(img1f2,11);
	sigma1_2 -= mu1_2;
	sigma2_2=gaussianBlur(img2f2,11);
	sigma2_2 -= mu2_2;
	sigma12=gaussianBlur(img1f_img2f,11);
	sigma12 -= mu1_mu2;

	Mat t1 = (2 * mu1_mu2 + C1) / (mu1_2 + mu2_2 + C1);
	Mat t2 = (2 * sigma12 + C2) / (sigma1_2 + sigma2_2 + C2);

	Mat ssim_map;
	multiply(t1, t2, ssim_map);

	Scalar mssim = mean(ssim_map);
	return mssim.val[0];
}


void ssim(){
	char fname[MAX_PATH];
	char fname2[MAX_PATH];
	while (openFileDlg(fname) && (openFileDlg(fname2)))
	{
		Mat src1 = imread(fname);
		Mat src2 = imread(fname2);
		cv::Mat resizedImg1, resizedImg2;
		//cv::resize(src1, resizedImg1, cv::Size(640, 480)); // new size: 640x480
		cv::resize(src2, resizedImg2, cv::Size(src1.cols, src1.rows)); // new size: 800x600

		double ssim = ssimCalc(src1, resizedImg2);
		
		cout << ssim << " ";
	}
}

double mseCalc(Mat i1, Mat i2) {
	int s = 0;
	for (int i = 0; i < i1.rows; i++) {
		for (int j = 0; j < i1.cols; j++) {
			s += pow(abs(i1.at<uchar>(i, j) - i2.at<uchar>(i, j)), 2);
		}
	}
	return s/(double)(i1.rows*i1.cols);
}
void mse() {
	char fname[MAX_PATH];
	char fname2[MAX_PATH];
	while (openFileDlg(fname) && (openFileDlg(fname2)))
	{
		Mat src1 = imread(fname, IMREAD_GRAYSCALE);
		Mat src2 = imread(fname2, IMREAD_GRAYSCALE);
		cv::Mat resizedImg1, resizedImg2;
		//cv::resize(src1, resizedImg1, cv::Size(640, 480)); // new size: 640x480
		cv::resize(src2, resizedImg2, cv::Size(src1.cols, src1.rows)); // new size: 800x600

		double sim = mseCalc(src1, resizedImg2);
		cout << sim << " ";
	}
}
int* getHistogram(Mat src, int nrBins) {
	int* h;
	h = (int*)calloc(nrBins, sizeof(int));
	int width = src.cols;
	int height = src.rows;
	int interval = 256 / nrBins;
	int k = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			 k =(int) (src.at<float>(i, j) / interval);
			 if (k < 0) {
				 h[0]++;
			 }
			 else if (k > 255) {
				 h[255]++;
			 }
			else h[k]++;
		}
	}
	return h;
}

int* prag(float* fdp, int wh, float th, int* p) {
	int* v = (int*)calloc(256, sizeof(int));
	int p1 = 1;
	for (int k = wh; k < 256 - wh; k++) {
		int ismx = 1;
		int index = 0;
		float m = 0;

		for (int i = k - wh; i <= k + wh; i++) {
			m += fdp[i];
			if (fdp[i] > fdp[k])
			{
				ismx = 0;
			}
		}
		m /= (float)(2 * wh + 1);

		if (ismx && (fdp[k] > (m + th))) {
			v[p1] = k;
			p1++;
		}

	}
	v[0] = 0;
	v[p1++] = 255;
	*p = p1;
	return v;
}

Mat pragmultiplu(Mat src,double th) {
	int* h;
	float* fdp;

		h = (int*)malloc(256 * sizeof(int));
		fdp = (float*)calloc(256, sizeof(float));
		h = getHistogram(src, 256);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_32F);
		for (int k = 0; k < 256; k++) {
			fdp[k] = (float)h[k] / (height * width);
		}
		int nrp = 0;
		//int* mx = (int*)calloc(256, sizeof(int));
		int* mx = prag(fdp, 5, th, &nrp);
		
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float val = src.at<float>(i, j);
				int mn = 999;
				int vec = 0;
				for (int k = 0; k < nrp; k++) {
					if (abs(val - mx[k]) < mn)
					{
						mn = val - mx[k];
						vec = mx[k];
					}
				}
				dst.at<float>(i, j) = vec;
			}
		}
		return dst;
	
}

Mat thresholdv2( Mat src, double threshold1) {
	Mat dst= Mat(src.rows,src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<float>(i, j) > threshold1) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}
bool isInside(int i, int j, int rows, int cols) {
	if (i >= 0 && i < rows && j >= 0 && j < cols)
		return true;
	return false;
}
Mat dilatare(Mat src1) {

	int height = src1.rows;
		int width = src1.cols;
		int di[] = { -1, 0, 0, 1 };
		int dj[] = { 0, -1, 1, 0 };

		int strElem[3][3] = { { 0,1,0}, { 1,1,1 }, { 0,1,0 } };

		Mat dst1 = Mat(height, width, CV_8UC1);
		dst1.setTo(255);

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {
				if (src1.at<uchar>(i, j) == 0) {
					dst1.at<uchar>(i, j) = 0;
					for (int k = 0; k < 4; k++)
						if (isInside(i + di[k], j + dj[k], height, width)) {
							dst1.at<uchar>(i + di[k], j + dj[k]) = 0;
						}
				}
			}

		return dst1;
}
Mat eroziune(Mat src1) {


		int height = src1.rows;
		int width = src1.cols;
		int di[] = { -1, 0, 0, 1 };
		int dj[] = { 0, -1, 1, 0 };

		int strElem[3][3] = { { 0,1,0}, { 1,1,1 }, { 0,1,0 } };

		Mat dst1 = Mat(height, width, CV_8UC1);
		dst1.setTo(255);

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {
				if (src1.at<uchar>(i, j) == 0) {
					int cont = 0;
					for (int k = 0; k < 4; k++)
						if (isInside(i + di[k], j + dj[k], height, width)) {
							if (src1.at<uchar>(i + di[k], j + dj[k]) == 0)
								cont++;
						}
					if (cont == 4)
						dst1.at<uchar>(i, j) = 0;

				}
			}

		return dst1;
}
Mat filtrarea2(Mat src, int nucleu[3][3]) {
	int di[] = { -1,-1,-1,0, 0, 0,1,1,1 };
	int dj[] = { -1,0, 1,-1, 0, 1,-1,0,1 };
	int height = src.rows;
	int width = src.cols;
	Mat dst2 = Mat(src.rows, src.cols, CV_32FC1);
	for (int i = 1; i < height - 1; i++)
		for (int j = 1; j < width - 1; j++) {
			int suma = 0;
			for (int k = 0; k < 9; k++)
				if (isInside(i + di[k], j + dj[k], height, width)) {
					suma += src.at<float>(i + di[k], j + dj[k]) * nucleu[1 + di[k]][1 + dj[k]];
				}
			dst2.at<float>(i, j) = suma;
		}
	return dst2;
}
Mat filtrarea(Mat src, int nucleu[3][3]) {
	int di[] = { -1,-1,-1,0, 0, 0,1,1,1 };
	int dj[] = { -1,0, 1,-1, 0, 1,-1,0,1 };
	int height = src.rows;
	int width = src.cols;
	Mat dst2 = Mat(src.rows, src.cols, CV_32FC1);
	for (int i = 1; i < height - 1; i++)
		for (int j = 1; j < width - 1; j++) {
			int suma = 0;
			for (int k = 0; k < 9; k++)
				if (isInside(i + di[k], j + dj[k], height, width)) {
					suma += src.at<uchar>(i + di[k], j + dj[k]) * nucleu[1 + di[k]][1 + dj[k]];
				}
			dst2.at<float>(i, j) = suma;
		}
	return dst2;
}

vector<cv::Point2f> harrisCorner(Mat image)
{


	int ksize = 3;
	
	int filtru1[3][3] = { { -1,0,1 }, { -2,0,2 }, { -1,0,1 } };
	int filtru2[3][3] = { { 1, 2,  1}, {  0, 0,  0 }, { -1, -2, -1} };

	Mat dx = filtrarea(image, filtru1);
	Mat dy = filtrarea (image, filtru2);


	Mat Ixx, Ixy, Iyy;
	multiply(dx, dx, Ixx);
	multiply(dx, dy, Ixy);
	multiply(dy, dy, Iyy);


	Mat Sxx, Sxy, Syy;
	Sxx = gaussianBlur(Ixx, 7);
	Sxy = gaussianBlur(Ixy, 7);
	Syy = gaussianBlur(Iyy, 7);


	double k = 0.04;
	Mat R(image.size(), CV_32F);
	double max_response = 0.0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float sxx = Sxx.at<float>(i, j);
			float sxy = Sxy.at<float>(i, j);
			float syy = Syy.at<float>(i, j);
			float det = sxx * syy - sxy * sxy;
			float trace = sxx + syy;
			R.at<float>(i, j) = det - k * trace * trace;
			if (R.at<float>(i, j) > max_response) {
				max_response = R.at<float>(i, j);
			}
		}
	}

	double threshold1 = 0.1 * max_response;
	Mat corners = thresholdv2(R, threshold1);
	imshow("output", corners);
	waitKey(0);
		Mat binaryImg = dilatare(corners);
		Mat corners2 = eroziune(binaryImg);
		std::vector<cv::Point2f> cornerPts;
		for (int i = 0; i < corners2.rows; i++) {
			for (int j = 0; j < corners2.cols; j++) {
				if (corners2.at<uchar>(i, j) == 255) {
					cornerPts.push_back(cv::Point2f(j, i));
				}
			}
		}
	return cornerPts;
}


Mat getRectSubPix(const cv::Mat& img, const cv::Size& patchSize, const cv::Point2f& center) {
	int patchWidth = patchSize.width;
	int patchHeight = patchSize.height;
	Mat patch;
	patch.create(patchHeight, patchWidth, img.type());

	// Compute the coordinates of the top-left corner of the patch
	float x1 = center.x - patchWidth / 2.0f;
	float y1 = center.y - patchHeight / 2.0f;

	// Iterate over each pixel in the patch and copy its value from the image
	for (int y = 0; y < patchHeight; y++) {
		for (int x = 0; x < patchWidth; x++) {
			// Compute the coordinates of the pixel in the image
			float x2 = x1 + x;
			float y2 = y1 + y;

			// If the pixel is outside of the image, set it to zero
			if (x2 < 0 || x2 >= img.cols || y2 < 0 || y2 >= img.rows) {
				patch.at<float>(y, x) = 0.0f;
			}
			else {
				// Otherwise, copy the pixel value from the image
				patch.at<float>(y, x) = img.at<float>(cv::Point2f(x2, y2));
			}
		}
	}
	return patch;
}


void computeLocalDescriptor(const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, std::vector<std::vector<float>>& descriptors) {
	const int patchSize = 16;
	const int patchRadius = patchSize / 2;
	const int numBins = 8;
	const float binSize = 360.f / numBins;

	descriptors.resize(keypoints.size(), std::vector<float>(numBins * numBins));

	for (int k = 0; k < keypoints.size(); k++) {
		const cv::Point2f& pt = keypoints[k];
		std::vector<float>& desc = descriptors[k];

		// Compute the gradient magnitude and orientation for a circular patch around the keypoint
		for (int i = -patchRadius; i <= patchRadius; i++) {
			for (int j = -patchRadius; j <= patchRadius; j++) {
				int y = std::round(pt.y + i);
				int x = std::round(pt.x + j);

				if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
					float dx = img.at<float>(y, max(x - 1, 0)) - img.at<float>(y,min(x + 1, img.cols - 1));
					float dy = img.at<float>(max(y - 1, 0), x) - img.at<float>(min(y + 1, img.rows - 1), x);
					float mag = std::sqrt(dx * dx + dy * dy);
					float ori = std::atan2(dy, dx) * 180.f / CV_PI;

					// Map the orientation to a bin index and add the magnitude to the corresponding bin
					int binX = std::floor((ori + 180.f) / binSize);
					int binY = std::floor((mag / 255.f) * numBins);

					if (binX >= numBins) {
						binX = numBins - 1;
					}
					if (binY >= numBins) {
						binY = numBins - 1;
					}

					desc[binY * numBins + binX] += mag;
				}
			}
		}

		// Normalize the descriptor
		float norm = 0.f;
		for (int i = 0; i < desc.size(); i++) {
			norm += desc[i] * desc[i];
		}

		norm = std::sqrt(norm);

		for (int i = 0; i < desc.size(); i++) {
			desc[i] /= norm;
		}
	}
}


void computeLocalDescriptor2(const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, std::vector<std::vector<float>>& descriptors) {
	const int patchSize = 16;
	const int patchRadius = patchSize / 2;
	const int numBins = 8;
	const float binSize = 360.f / numBins;
	
	descriptors.resize(keypoints.size(), std::vector<float>(numBins * numBins));
	int filtru1[3][3] = { { -1,0,1 }, { -2,0,2 }, { -1,0,1 } };
	int filtru2[3][3] = { { 1, 2,  1}, {  0, 0,  0 }, { -1, -2, -1} };

	

	for (int k = 0; k < keypoints.size(); k++) {
		const cv::Point2f& pt = keypoints[k];
		std::vector<float>& desc = descriptors[k];

		Mat patch=getRectSubPix(img, cv::Size(patchSize, patchSize), pt);

		Mat Ix = filtrarea2(patch, filtru1);
		Mat Iy = filtrarea2(patch, filtru2);
		for (int i = 0; i < patch.rows; i++) {
			for (int j = 0; j < patch.cols; j++) {
				float dx = Ix.at<float>(i, j);
				float dy = Iy.at<float>(i, j);
				float mag = std::sqrt(dx * dx + dy * dy);
				float val = atan2(dy, dx);
				float ori=val;
				/*if (val >= 0)
				  ori = val* 180.f / CV_PI;
				else ori= (val+2*CV_PI) * 180.f / CV_PI;
				*/
				// Map the orientation to a bin index and add the magnitude to the corresponding bin
				int binX = std::floor((ori + 180.f) / binSize);
				int binY = std::floor((mag / 255.f) * numBins);

				if (binX >= numBins) {
					binX = numBins - 1;
				}
				if (binY >= numBins) {
					binY = numBins - 1;
				}

				desc[binY * numBins + binX] += mag;
			}
		}

		// Normalize the descriptor
		float norm = 0.f;
		for (int i = 0; i < desc.size(); i++) {
			norm += desc[i] * desc[i];
		}

		norm = std::sqrt(norm);

		for (int i = 0; i < desc.size(); i++) {
			desc[i] /= norm;
		}
	}

}


float euclideanDist(const std::vector<float>& v1, const std::vector<float>& v2)
{
	float dist = 0.0f;
	for (size_t i = 0; i < v1.size(); i++) {
		float diff = v1[i] - v2[i];
		dist += diff * diff;
	}
	return sqrt(dist);
}


std::vector<cv::DMatch> harrisMatch(std::vector < std::vector<float>> descriptors1, std::vector < std::vector<float>> descriptors2){

	const int k = 2; 

	
	std::vector<cv::DMatch> matches;

	
	for (int i = 0; i < descriptors1.size(); i++) {
		
		float bestDistances[k] = { FLT_MAX,FLT_MAX };
		int bestIndices[k] = { -1, -1 };

		
		for (int j = 0; j < descriptors2.size(); j++) {
			
			float dist = 0;
			for (int d = 0; d < descriptors1[i].size(); d++) {
				float diff = descriptors1[i][d] - descriptors2[j][d];
				dist += diff * diff;
			}

			
			if (dist < bestDistances[0]) {
				bestDistances[1] = bestDistances[0];
				bestIndices[1] = bestIndices[0];
				bestDistances[0] = dist;
				bestIndices[0] = j;
			}
			else if (dist < bestDistances[1]) {
				bestDistances[1] = dist;
				bestIndices[1] = j;
			}
		}


		float distanceRatio = bestDistances[0] / bestDistances[1];
		if (distanceRatio < 0.8) {
			
			matches.push_back(cv::DMatch(i, bestIndices[0], bestDistances[0]));
		}
	}
	return matches;
}



void harris() {
	char fname[MAX_PATH];
	char fname2[MAX_PATH];
	while (openFileDlg(fname) && (openFileDlg(fname2)))
	{
		Mat src1 = imread(fname, IMREAD_GRAYSCALE);
		Mat src2 = imread(fname2, IMREAD_GRAYSCALE);
	
		cv::Mat floatImage1,floatImage2;
		src1.convertTo(floatImage1, CV_32FC1);
		src2.convertTo(floatImage2, CV_32FC1);
		vector<cv::Point2f> keypoints1 = harrisCorner(src1);
		vector<cv::Point2f> keypoints2 = harrisCorner(src2);
		std::vector < std::vector<float>> des2, des1;
		 computeLocalDescriptor2(floatImage1, keypoints1,des1);
		 computeLocalDescriptor2(floatImage2, keypoints2,des2);
		
		 std::vector<cv::DMatch> matches;
		 float size1;
		 if (des1.size() > des2.size()) {
			  matches = harrisMatch(des1, des2);
			  size1 = 100 * matches.size() / (float)des1.size();
		 }
		 else {
			 matches = harrisMatch(des2, des1);
			 size1 = 100 * matches.size() / (float)des2.size();
		 }
		
		
		

		std::cout << "Number of good matches: " << matches.size() << std::endl;
		//float size1 = 100*matches.size() /(float)des1.size();
		printf("there are %f good matches\n", size1);
		
		
	}
}

void orb_matching() {
	char fname[MAX_PATH];
	char fname2[MAX_PATH];
	while (openFileDlg(fname)&&openFileDlg(fname2)) {

			Mat img1 = imread(fname, IMREAD_GRAYSCALE);         
			Mat img2 = imread(fname2, IMREAD_GRAYSCALE); 

			float lowe_ratio = 0.89;

			Ptr<Feature2D> orb;
		
				orb = ORB::create();
			
			

				std::vector<KeyPoint> kp1, kp2;
				Mat des1, des2;
				orb->detectAndCompute(img1, noArray(), kp1, des1);
				orb->detectAndCompute(img2, noArray(), kp2, des2);

			
				BFMatcher matcher(NORM_HAMMING, true);
			    vector<cv::DMatch> knn_matches;
				matcher.match(des1, des2, knn_matches);
			
				std::vector<cv::DMatch> good;
				for (size_t i = 0; i < knn_matches.size()-1; i++) {
					if (knn_matches[i].distance < lowe_ratio * knn_matches[i + 1].distance) {
						good.push_back(knn_matches[i]);
					}
				}

			String msg2 = format("there are %d good matches", good.size());

			Mat img3;
			drawMatches(img1, kp1, img2, kp2, good, img3, Scalar::all(-1),
				Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			
			putText(img3, msg2, Point(10, 270), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
			printf("there are % d good matches", good.size());
			String fname = format("output_ORB_%.2f.png", lowe_ratio);
			imwrite(fname, img3);

			imshow("Matches", img3);
			waitKey(0);
		



	}
}
void orb_matching2() {
	char fname[MAX_PATH];
	char fname2[MAX_PATH];
	while (openFileDlg(fname) && openFileDlg(fname2)) {

		Mat img11 = imread(fname);          
		Mat img21 = imread(fname2); 
		Mat hsv,hsv2;
		Mat img1, img2;
		cvtColor(img11, hsv, COLOR_BGR2HSV);

		
		std::vector<Mat> hsv_channels;
		split(hsv, hsv_channels);

		
		equalizeHist(hsv_channels[2], hsv_channels[2]);

		
		merge(hsv_channels, hsv);

		Mat equalized_img;
		cvtColor(hsv, img1, COLOR_HSV2BGR);

		cvtColor(img21, hsv2, COLOR_BGR2HSV);

		std::vector<Mat> hsv_channels2;
		split(hsv2, hsv_channels2);

		equalizeHist(hsv_channels2[2], hsv_channels2[2]);

		
		merge(hsv_channels2, hsv2);

		cvtColor(hsv2, img2, COLOR_HSV2BGR);

		
		
		float lowe_ratio = 0.89;

		Ptr<Feature2D> orb;

		orb = ORB::create();



		std::vector<KeyPoint> kp1, kp2;
		Mat des1, des2;
		orb->detectAndCompute(img1, noArray(), kp1, des1);
		orb->detectAndCompute(img2, noArray(), kp2, des2);

	
		BFMatcher matcher(NORM_HAMMING, true);
		vector<cv::DMatch> knn_matches;
		matcher.match(des1, des2, knn_matches);

		std::vector<cv::DMatch> good1;
		for (size_t i = 0; i < knn_matches.size() - 1; i++) {
			if (knn_matches[i].distance < lowe_ratio * knn_matches[i + 1].distance) {
				good1.push_back(knn_matches[i]);
			}
		}

		vector<cv::DMatch> knn_matches2;
		matcher.match(des2, des1, knn_matches2);

		std::vector<cv::DMatch> good2;
		for (size_t i = 0; i < knn_matches2.size() - 1; i++) {
			if (knn_matches2[i].distance < lowe_ratio * knn_matches2[i + 1].distance) {
				good2.push_back(knn_matches2[i]);
			}
		}
		vector<DMatch> topResults;
		for (const auto& match1 : good1) {
			int match1QueryIndex = match1.queryIdx;
			int match1TrainIndex = match1.trainIdx;

			for (const auto& match2 : good2) {
				int match2QueryIndex = match2.queryIdx;
				int match2TrainIndex = match2.trainIdx;

				if ((match1QueryIndex == match2TrainIndex) && (match1TrainIndex == match2QueryIndex)) {
					topResults.push_back(match1);
				}
			}
		}
		String msg2 = format("there are %d good matches", topResults.size());

		Mat img3;
		drawMatches(img11, kp1, img21, kp2, topResults, img3, Scalar::all(-1),
			Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


		putText(img3, msg2, Point(10, 270), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
		printf("there are % d good matches", topResults.size());
		String fname = format("output_ORB_%.2f.png", lowe_ratio);
		imwrite(fname, img3);

		imshow("Matches", img3);
		waitKey(0);




	}
}
void harrisCorner2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat image = imread(fname, IMREAD_GRAYSCALE);



		int ksize = 3;
		Mat dx, dy;
		Sobel(image, dx, CV_32F, 1, 0, ksize);
		Sobel(image, dy, CV_32F, 0, 1, ksize);


		Mat Ixx, Ixy, Iyy;
		multiply(dx, dx, Ixx);
		multiply(dx, dy, Ixy);
		multiply(dy, dy, Iyy);


		Mat Sxx, Sxy, Syy;
		Sxx = gaussianBlur(Ixx, 7);
		Sxy = gaussianBlur(Ixy, 7);
		Syy = gaussianBlur(Iyy, 7);


		double k = 0.04;
		Mat R(image.size(), CV_32F);
		double max_response = 0.0;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				float sxx = Sxx.at<float>(i, j);
				float sxy = Sxy.at<float>(i, j);
				float syy = Syy.at<float>(i, j);
				float det = sxx * syy - sxy * sxy;
				float trace = sxx + syy;
				R.at<float>(i, j) = det - k * trace * trace;
				if (R.at<float>(i, j) > max_response) {
					max_response = R.at<float>(i, j);
				}
			}
		}

		// Threshold the Harris corner response to obtain corners
		double threshold1 = 0.1 * max_response;
		Mat corners= thresholdv2(R, threshold1);
		//threshold(R, corners, threshold1, 255, THRESH_BINARY);
		//corners = pragmultiplu(R, threshold1);
		// Display the corners
		namedWindow("Corners", WINDOW_NORMAL);
		imshow("Corners", corners);
		waitKey();
	}
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf("13-mse\n");
		printf("14-gaussian\n");
		printf("15-ssim\n");
		printf("16-harris corner\n");
		printf("17-orb \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				mse();
				break;
			case 14:
				gaussianFilter();
				break;
			case 15:
				ssim();
				break;
			case 16:
				harris();
				break;
			case 17:
				orb_matching2();
				break;
		}
	}
	while (op!=0);
	return 0;
}