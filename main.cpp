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
	double xs = 0;
	double ys = 0;
	double g = 0;
	double x = 0;
	double y = 0;
	void set(double xs1 = 0, double ys1 = 0, double g1=0, double x1=0, double y1=0){
		xs=xs1; ys=ys1; g=g1; x=x1; y=y1;
	}
};

class featureSpace {
public:
	featureSpace();
	featureSpace(Mat&, int, double);
	double mean_shift(int X, int Y);
	double kernel(int X, int Y, int Xc, int Yc);
	double getGrayValue(int X, int Y);		//get gray value at (X,Y)
	bool isConvergent(fsPoint, fsPoint);
private:
	Mat gray_value;
	int rows;
	int cols;
	int hs;
	double hr;
};

featureSpace::featureSpace():rows(0),cols(0){}
featureSpace::featureSpace(Mat& gv, int hspatial, double hrange) {
	// warning: do not forget normalization!!!
	gray_value = gv.clone();
	//gray_value.convertTo(gray_value, CV_32F);	//convert to float for normalization
	rows = gray_value.rows;
	cols = gray_value.cols;
	//gray_value /= 256;
	hs = hspatial;
	hr = hrange;
}

double featureSpace::mean_shift(int X, int Y) {
	// compute convergence point of (X,Y), and return convergence gray value;
	// compute m(x)
	int counter = 0;
	double tmpU = 0; //temp for dominator
	fsPoint xp;	//x prime
	fsPoint xf; //x former
	xf.set(X, Y);
	xf.g = getGrayValue(X, Y);
	xf.x = X*2;		//for normalizing to 0-256
	xf.y = Y*2;
	while(1){
		// set ROI for speeding up calculation
		int xmin = xf.xs - hs > 0 ? (xf.xs - hs) : 0;
		int xmax = xf.xs + hs < cols ? (xf.xs+ hs) : cols;
		int ymin = xf.ys - hs > 0 ? (xf.ys - hs) : 0;
		int ymax = xf.ys + hs < rows ? (xf.ys + hs) : rows;
		counter++;
		for (int i = xmin;i < xmax;i++) {		 //optimization: to reduce iteration times
			//uchar* g = gray_value.ptr<uchar>(i); //pointor to gray value
			for (int j = ymin;j < ymax;j++) {
				xp.x += i*kernel(i, j, xf.x, xf.ys);
				xp.y += j*kernel(i, j, xf.x, xf.ys);
				tmpU += kernel(i, j, xf.xs, xf.ys);
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
		xf.x = xp.x;
		xf.y = xp.y;
		//xp.g = 0;
		xp.x = 0;
		xp.y = 0;
		tmpU = 0;
	}
	//cout<<counter<<" [ "<<xp.x/2<<", "<<xp.y/2<<" ]"<<endl;
	return getGrayValue(xp.x/2,xp.y/2);
	//return xp.g*(256 / rows);
	
}

double featureSpace::kernel(int X, int Y, int Xc, int Yc) {
	// use Epanechnikov kernal
	// kernel only applied to gray value channel!!
	//double dist = abs(getGrayValue(X, Y) - getGrayValue(Xc, Yc));
	//cout<<"pl1: "<<X<<" "<<Y<<" "<<Xc<<" "<<Yc<<endl;
	/*
	fsPoint tmp;
	tmp.x = 2.0*(X - Xc)/hr;
	tmp.y = 2.0*(Y - Yc)/hr;
	tmp.g = 1.0*(getGrayValue(X, Y) - getGrayValue(Xc, Yc))/hr;
	double dist = sqrt(tmp.x*tmp.x + tmp.y*tmp.y + tmp.g*tmp.g);
	*/
	
	double dist = sqrt(1.0*(X - Xc)*(X - Xc) + 
						1.0*(Y - Yc)*(Y - Yc)+
						1.0*(getGrayValue(X, Y) - getGrayValue(Xc, Yc))*(getGrayValue(X, Y) - getGrayValue(Xc, Yc))
					  );
	if (dist < hr) return 1.0;
	//cout<< "kernel calculation success"<<endl;

	//if (dist < 1) return 1.5*dist;
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
	//tmp.g = xp.g - xf.g;
	tmp.x = xp.x - xf.x;
	tmp.y = xp.y - xf.y;
//	if (sqrt(tmp.g*tmp.g + tmp.x*tmp.x + tmp.y*tmp.y) < delta)
	if (sqrt(tmp.x*tmp.x + tmp.y*tmp.y) < delta)
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
	cout << "gray-value:" << endl << img << endl;
	
	featureSpace fs(img, 8, 32);
	Mat denoiseImg(img.rows, img.cols, img.type());
	// apply mean-shift
	for (int i = 0;i < img.cols;i++) {
		uchar* data = denoiseImg.ptr<uchar>(i);
		for (int j = 0;j < img.rows;j++) {
			data[j] = (uchar)fs.mean_shift(i, j);
		}
	}
	namedWindow("denoised", WINDOW_NORMAL);
	//namedWindow("denoised", WINDOW_AUTOSIZE);
	imshow("denoised", denoiseImg);
	

	//cout << img << endl;
	waitKey(0);
	destroyAllWindows();
	return 0;
}
