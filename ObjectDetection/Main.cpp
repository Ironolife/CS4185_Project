/**
* CS4185/CS5185 Multimedia Technologies and Applications
* Course Assignment
* Object Detection Task
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/nonfree/features2d.hpp"  

#include <iostream>
#include <stdio.h>
#include <algorithm>

using namespace std;
using namespace cv;

#define IMAGE_folder "D:\\dataset"  // change to your folder location
#define IMAGE_LIST_FILE "dataset2"      //the dataset2 for detection
#define DETECTION_IMAGE 1               //change from 1 to 10 as the detection images to get your output
#define SEARCH_IMAGE "football.png"     //input information

#define     DBL_MAX      1.7976931348623158e+308

/**
* @function main
*/

//Compute pixel-by-pixel difference
double compareImgs(Mat img1, Mat img2)
{
	int w = img1.cols, h = img1.rows;
	int size = w * h;
	Mat new_img2;
	resize(img2, new_img2, img1.size());
	double sum = 0;
	for (int i = 0; i < w; i++)for (int j = 0; j < h; j++)
	{
		sum += abs(img1.at<uchar>(j, i) - new_img2.at<uchar>(j, i));
	}
	return sum / 256 / size;
}

int main(int argc, char** argv)
{
	Mat src_input, gray_input;
	Mat db_img, db_gray_img;

	const int filename_len = 900;
	char tempname[filename_len];

	const int score_size = 10;  //top n match regions
	double score[score_size] = { DBL_MAX };
	int start_x[score_size], start_y[score_size], end_x[score_size], end_y[score_size];  //store the start location and end location of the detection region (bounding box)

	sprintf_s(tempname, filename_len, "%s\\%s", IMAGE_folder, SEARCH_IMAGE);
	src_input = imread(tempname); // read input image
	if (!src_input.data)
	{
		printf("Cannot find the input image!\n");
		system("pause");
		return -1;
	}
	imshow("Input", src_input);
	// Convert to grayscale
	cvtColor(src_input, gray_input, COLOR_BGR2GRAY);


	//Read detection IMAGE
	sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, DETECTION_IMAGE);
	db_img = imread(tempname); // read besearched image
	if (!db_img.data)
	{
		printf("Cannot find the detection image number!\n");
		system("pause");
		return -1;
	}
	cvtColor(db_img, db_gray_img, COLOR_BGR2GRAY);     // Convert to grayscale


													   //search the image by bouding box from diffrent scale, location, length-width ratio
	int w = db_gray_img.cols, h = db_gray_img.rows;
	//the starting scale in the search, you can change it to a smaller or larger scale
	int scale_w = 20, scale_h = 20;
	//the ending scale in the search, you can change it to a smaller or larger scale
	int max_scale_w = 150, max_scale_h = 150;
	//you can change the search step of scale and location in your code, 
	//which will influce both the perforance and speed, you may need a tradeoff
	int scale_step = 5, location_step = 5;

	//we assume the scale_w should be equals to scale_h in the round ball detection, 
	//thus length-width ratio is always 1 in this algorithmn.
	//For other object, you may need to try different length-width ratio  
	for (scale_w; scale_w<max_scale_w; scale_w += scale_step, scale_h += scale_step) {
		int max_x = w - scale_w, max_y = h - scale_h;
		for (int x = 0; x < max_x; x += location_step) for (int y = 0; y < max_y; y += location_step)
		{
			//capture the image region in the searching bounding box
			Mat db_region_img(db_gray_img, Rect(x, y, scale_w, scale_h));
			// Apply the pixel-by-pixel comparison method
			double tempScore = compareImgs(db_region_img, gray_input);

			//store the top k(k=score_size) match bounding box and score
			for (int k = 0; k<score_size; k++) {
				if (tempScore < score[k]) {
					for (int k1 = score_size - 1; k1>k; k1--) {
						score[k1] = score[k1 - 1];
						start_x[k1] = start_x[k1 - 1];
						start_y[k1] = start_y[k1 - 1];
						end_x[k1] = end_x[k1 - 1];
						end_y[k1] = end_y[k1 - 1];
					}
					score[k] = tempScore;
					start_x[k] = x;
					start_y[k] = y;
					end_x[k] = x + scale_w;
					end_y[k] = y + scale_h;
					break;
				}
			}
		}
	}

	// draw the best match[top k (k=score_size)] rectangele
	for (int k = 0; k<score_size; k++) {
		Point start = Point(start_x[k], start_y[k]);
		Point end = Point(end_x[k], end_y[k]);
		rectangle(db_img, start, end, Scalar(255, 0, 0));
	}

	//you should keep this evalation code unchanged: 
	//compare your detection boulding box with ground truth bouding box by IoU 
	//first we define the location of ground truth bouding box
	const int gt_start_x[10] = { 266,220,200,238,350,26,204,128,33,380 };
	const int gt_start_y[10] = { 146,248,83,120,80,10,347,258,196,207 };
	const int gt_end_x[10] = { 353,380,324,314,391,78,248,156,75,404 };
	const int gt_end_y[10] = { 233,398,207,196,121,62,391,288,238,231 };
	//draw ground truth bouding box
	Point start = Point(gt_start_x[DETECTION_IMAGE - 1], gt_start_y[DETECTION_IMAGE - 1]);
	Point end = Point(gt_end_x[DETECTION_IMAGE - 1], gt_end_y[DETECTION_IMAGE - 1]);
	rectangle(db_img, start, end, Scalar(0, 0, 255));
	int gt_area = (gt_end_x[DETECTION_IMAGE - 1] - gt_start_x[DETECTION_IMAGE - 1]) * (gt_end_y[DETECTION_IMAGE - 1] - gt_start_y[DETECTION_IMAGE - 1]);
	//calculate top 10 IoU, and print the best one
	double best_IoU = 0;
	for (int k = 0; k<score_size; k++) {
		int intersect_start_x = start_x[k]>gt_start_x[DETECTION_IMAGE - 1] ? start_x[k] : gt_start_x[DETECTION_IMAGE - 1];
		int intersect_start_y = start_y[k]>gt_start_y[DETECTION_IMAGE - 1] ? start_y[k] : gt_start_y[DETECTION_IMAGE - 1];
		int intersect_end_x = end_x[k]<gt_end_x[DETECTION_IMAGE - 1] ? end_x[k] : gt_end_x[DETECTION_IMAGE - 1];
		int intersect_end_y = end_y[k]<gt_end_y[DETECTION_IMAGE - 1] ? end_y[k] : gt_end_y[DETECTION_IMAGE - 1];

		int your_area = (end_x[k] - start_x[k]) * (end_y[k] - start_y[k]);
		int intersect_area = 0;
		if (intersect_end_x > intersect_start_x && intersect_end_y > intersect_start_y) {
			intersect_area = (intersect_end_x - intersect_start_x) * (intersect_end_y - intersect_start_y);
		}
		int union_area = gt_area + your_area - intersect_area;
		double IoU = (double)intersect_area / union_area;
		if (IoU > best_IoU) {
			best_IoU = IoU;
		}
	}
	printf("The best IoU in your top 10 detection is %f\n", best_IoU);

	//show and store the detection reuslts
	imshow("Best Match Image", db_img);
	sprintf_s(tempname, filename_len, "%s\\detectionResults\\%d.jpg", IMAGE_folder, DETECTION_IMAGE);
	imwrite(tempname, db_img);

	printf("Done \n");
	// Wait for the user to press a key in the GUI window.
	//Press ESC to quit
	int keyValue = 0;
	while (keyValue >= 0)
	{
		keyValue = cvWaitKey(0);

		switch (keyValue)
		{
		case 27:keyValue = -1;
			break;
		}
	}

	return 0;
}
