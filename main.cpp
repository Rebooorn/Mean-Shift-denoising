#include <opencv2/opencv.hpp>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace cv;
using namespace std;

const double delta = 0.1;

struct fsPoint		
{
	//feature vector: spatial(xs,ys), range(g,x,y)
	double g = 0;
	double x = 0;
	double y = 0;
	void set(double g1=0, double x1=0, double y1=0){
		g=g1; x=x1; y=y1;
	}
};

class featureSpace {
public:
	featureSpace();
	featureSpace(Mat&, double);
	double mean_shift(int X, int Y);
	double kernel(fsPoint&, fsPoint&);
	double getGrayValue(int X, int Y);		//get gray value at (X,Y)
	bool isConvergent(fsPoint, fsPoint);
private:
	Mat gray_value;
	int rows;
	int cols;
	int hs;
	double hr;
	vector<fsPoint> fs;
};

featureSpace::featureSpace():rows(0),cols(0){}
featureSpace::featureSpace(Mat& gv, double hrange) {
	// warning: do not forget normalization!!!
	gray_value = gv.clone();
	//gray_value.convertTo(gray_value, CV_32F);	//convert to float for normalization
	rows = gray_value.rows;
	cols = gray_value.cols;
	//gray_value /= 256;
	//hs = hspatial;
	hr = hrange;
	//generate feature space
	fsPoint tmp;
	for (int i = 0; i < rows; i++) {
		uchar* data = gray_value.ptr<uchar>(i);
		for (int j = 0; j < cols; j++) {
			tmp.x = (double)j / cols;		//normalization
			tmp.y = (double)i / rows;
			tmp.g = 1.0*data[j] / 256;
			fs.push_back(tmp);
		}
	}
	cout << "feature space loaded successfully" << endl;
}

double featureSpace::mean_shift(int X, int Y) {
	// compute convergence point of (X,Y), and return convergence gray value;
	// compute m(x)
	int counter = 0;
	double tmpU = 0; //temp for dominator
	fsPoint xp;	//x prime
	fsPoint xf; //x former
	int tar = Y*cols + X;
	xf.g = fs[tar].g;
	xf.x = fs[tar].x;		//for normalizing to 0-256
	xf.y = fs[tar].y;
	while(1){
		int xmin = (int)round((xf.x - hr)*cols);	//use window to eliminate iteration times
		int xmax = (int)round((xf.x + hr)*cols);
		int ymin = (int)round((xf.y - hr)*rows);
		int ymax = (int)round((xf.y + hr)*rows);
		xmin = xmin < 0 ? 0 : xmin;
		xmax = xmax > cols ? cols : xmax;
		ymin = ymin < 0 ? 0 : ymin;
		ymax = ymax > rows ? rows : ymax;
		for (int i = xmin;i < xmax;i++) {
			for (int j = ymin;j < ymax;j++) {
				// scan point is j*cols+i
				xp.g += fs[j*cols + i].g*kernel(fs[j*cols + i], xf);
				xp.x += fs[j*cols + i].x*kernel(fs[j*cols + i], xf);
				xp.y += fs[j*cols + i].y*kernel(fs[j*cols + i], xf);
				tmpU += kernel(fs[j*cols + i], xf);
			}
		}
		/*
		for (int i = 0;i < fs.size();i++) {
			if (abs(fs[i].x - fs[tar].x) < hr && abs(fs[i].y - fs[tar].y) < hr) {
				xp.g += fs[i].g*kernel(fs[i], xf);
				xp.x += fs[i].x*kernel(fs[i], xf);
				xp.y += fs[i].y*kernel(fs[i], xf);
				tmpU += kernel(fs[i], xf);
			}
		}
		*/
		if (tmpU == 0) xp.set(xf.g, xf.x, xf.y);
		else {
			xp.g /= tmpU;
			xp.x /= tmpU;
			xp.y /= tmpU;
		}
		if (isConvergent(xp, xf) || counter > 10) break;
		//not convergent, assign xp to xf, xf to 0
		xf.set(xp.g, xp.x, xp.y);
		xp.set(0, 0, 0);
		tmpU = 0;
		/*
		// set ROI for speeding up calculation
		int xmin = xf.x - hs > 0 ? (xf.x - hs) : 0;
		int xmax = xf.x + hs < cols ? (xf.x + hs) : cols;
		int ymin = xf.y - hs > 0 ? (xf.y - hs) : 0;
		int ymax = xf.y + hs < rows ? (xf.y + hs) : rows;
		counter
		for (int i = xmin;i < xmax;i++) {		 //optimization: to reduce iteration times
			for (int j = ymin;j < ymax;j++) {
				xp.x += i*kernel(i, j, (int)xf.x, (int)xf.y);
				xp.y += j*kernel(i, j, (int)xf.x, (int)xf.y);
				tmpU += kernel(i, j, (int)xf.x, (int)xf.y);
			}
		}
		//xp.g /= tmpU;
		if(tmpU==0){
			xp.x=xf.x;
			xp.y=xf.y;
		}else{
			xp.x = xp.x / tmpU;
			xp.y = xp.y / tmpU;
		}
		if (isConvergent(xf, xp) || counter>40) break;
		// otherwise reload xf, erase xp
		//xf.g = xp.g;
		xf.x = round(xp.x);
		xf.y = round(xp.y);
		//xp.g = 0;
		xp.x = 0;
		xp.y = 0;
		tmpU = 0;
		*/
	}
	//cout << "pixel: " << tar << " completed" << endl;
	//cout << "pixel: " << tar << " completed:" << " [ " << xp.x*128 << ", " << xp.y*128 << " ]" << endl;
	//cout<<counter<<" [ "<<xp.x<<", "<<xp.y<<" ]"<<endl;
	return xp.g;
	
}

double featureSpace::kernel(fsPoint& X, fsPoint& Xc) {
	// use Epanechnikov kernal
	// kernel only applied to gray value channel!!
	//double dist = abs(getGrayValue(X, Y) - getGrayValue(Xc, Yc));
	//cout<<"pl1: "<<X<<" "<<Y<<" "<<Xc<<" "<<Yc<<endl;
	
	fsPoint tmp;
	tmp.set(X.g-Xc.g, X.x-Xc.x, X.y-Xc.y);
	double dist = sqrt(tmp.x*tmp.x + tmp.y*tmp.y + tmp.g*tmp.g);

	if (dist < hr) return 1.0;
	else return 0.0;
}

double featureSpace::getGrayValue(int X, int Y) {
	//return gray value of (X,Y)
	//cout<< "pl2: " <<X<<" "<<Y<<endl;
	uchar* data = gray_value.ptr<uchar>(X);
	return data[Y];
}

bool featureSpace::isConvergent(fsPoint xp, fsPoint xf) {
	//x prime and x former
	// if Euclidian distance minor than delta
	fsPoint tmp;
	tmp.x = xp.x - xf.x;
	tmp.y = xp.y - xf.y;
//	if (sqrt(tmp.g*tmp.g + tmp.x*tmp.x + tmp.y*tmp.y) < delta)
	if (sqrt(tmp.x*tmp.x + tmp.y*tmp.y + tmp.g*tmp.g) < delta)
		return true;	//minor than delta
	else return false;
}




//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	Mat img = imread("cameraman_noisy.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty()) {
		// image open failed
		cout << "Error: image cannot be opened......." << endl;
		return -1;
	}
	namedWindow("origin", WINDOW_NORMAL);
	//namedWindow("origin", WINDOW_AUTOSIZE);
	imshow("origin", img);
	//cout << "gray-value:" << endl << img << endl;
	
	featureSpace fs(img, 0.2);
	Mat denoiseImg(img.rows, img.cols, img.type());
	// apply mean-shift
	for (int i = 0;i < img.rows;i++) {
		uchar* data = denoiseImg.ptr<uchar>(i);
		for (int j = 0;j < img.cols;j++) {
			data[j] = (uchar)(fs.mean_shift(j, i)*256);

		}
	}
	cout << "denoise successful" << endl;
	namedWindow("denoised", WINDOW_NORMAL);
	//namedWindow("denoised", WINDOW_AUTOSIZE);
	imshow("denoised", denoiseImg);
	

	//cout << img << endl;
	waitKey(0);
	destroyAllWindows();
	return 0;
}
