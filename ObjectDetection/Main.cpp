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
#include <ctime>
#include <math.h>

using namespace std;
using namespace cv;

#define IMAGE_folder "H:\\dataset" // Change to your folder location
#define IMAGE_LIST_FILE "dataset2" // The dataset2 for detection
#define DETECTION_IMAGE 1 // Change from 1 to 10 as the detection images to get your output
#define SEARCH_IMAGE "football.png" // Input information

bool findPentagons(Mat img) {
	// Convert to grayscale
	Mat gray;
	cvtColor(img, gray, COLOR_RGB2GRAY);

	// Apply blur
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);

	Mat canny_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Detect edges using canny
	Canny(gray, canny_output, 150, 150 * 2, 3);
	// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>> contours_poly(contours.size());

	for (int i = 0; i < contours.size(); i++)
		approxPolyDP(Mat(contours[i]), contours_poly[i], arcLength(contours[i], true) * 0.04, true);

	// Filter for convex pentagons
	vector<vector<Point>> hull(contours_poly.size());
	for (int i = 0; i < contours_poly.size(); i++) {
		if (contours_poly[i].size() == 5) {
			convexHull(Mat(contours_poly[i]), hull[i], false);
		}
	}

	for (int i = 0; i < hull.size(); i++) {
		if (hull[i].size() == 5) {

			/*Mat drawing = Mat::zeros(gray.size(), CV_8UC3);

			for (int i = 0; i < contours_poly.size(); i++) {
			drawContours(drawing, hull, i, Scalar(255, 0, 0), 1, 8, vector<Vec4i>(), 0, Point());
			}

			namedWindow("Pentagons", CV_WINDOW_AUTOSIZE);
			imshow("Pentagons", drawing);

			waitKey(0);*/

			return true;

		}
	}

	return false;
}

Mat getHistogram(Mat img)
{
	// Convert to HSV
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	// Calculate histogram
	int hBins = 8, sBins = 12;
	int histSize[] = { hBins, sBins };
	float hRanges[] = { 0, 180 }, sRanges[] = { 0, 256 };
	const float* ranges[] = { hRanges, sRanges };
	int channels[] = { 0, 1 };

	Mat histogram;

	for (int i = 0; i < 5; i++) {
		calcHist(&hsv, 1, channels, Mat(), histogram, 2, histSize, ranges);
		normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());
	}

	return histogram;
}

// Compute pixel-by-pixel difference
double compareImgs(Mat img1, Mat img2)
{
	double score = compareHist(getHistogram(img2), getHistogram(img1), 1);

	if (findPentagons(img1)) {
		score = 0;
	}

	return score;
}

int main(int argc, char** argv)
{
	std::clock_t startTime;

	Mat src_input, db_img;

	const int filename_len = 900;
	char tempname[filename_len];

	const int score_size = 10; // Top n match regions
	double score[score_size] = { DBL_MAX };
	int start_x[score_size], start_y[score_size], end_x[score_size], end_y[score_size]; // Store the start location and end location of the detection region (bounding box)

	sprintf_s(tempname, filename_len, "%s\\%s", IMAGE_folder, SEARCH_IMAGE);
	src_input = imread(tempname); // Read input image
	if (!src_input.data)
	{
		printf("Cannot find the input image!\n");
		system("pause");
		return -1;
	}
	imshow("Input", src_input);

	// Read detection IMAGE
	sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, DETECTION_IMAGE);
	db_img = imread(tempname); // read besearched image
	if (!db_img.data)
	{
		printf("Cannot find the detection image number!\n");
		system("pause");
		return -1;
	}

	// Search the image by bouding box from diffrent scale, location, length-width ratio
	int w = db_img.cols, h = db_img.rows;
	// The starting scale in the search, you can change it to a smaller or larger scale
	int scale_w = 30, scale_h = 30;
	// The ending scale in the search, you can change it to a smaller or larger scale
	int max_scale_w = 150, max_scale_h = 150;
	// You can change the search step of scale and location in your code, 
	// Which will influce both the perforance and speed, you may need a tradeoff
	int scale_step = 10;

	// Input for visualization
	bool visualize; char visC;
	Mat visImg;
	printf("Visualize comparison process? (Y/N): ");
	cin >> visC;
	if (visC == 'Y') {
		visualize = true;
		db_img.copyTo(visImg);
		printf("Press any key when focusing on the visualization window to go to next scale.\n");
	}
	else {
		visualize = false;
	}

	double areaScore[999] = {-1};

	startTime = std::clock();

	// We assume the scale_w should be equals to scale_h in the round ball detection, 
	// Thus length-width ratio is always 1 in this algorithmn.
	// For other object, you may need to try different length-width ratio  
	for (scale_w; scale_w < max_scale_w; scale_w += scale_step, scale_h += scale_step) {

		int location_step = scale_w;

		int max_x = w - scale_w, max_y = h - scale_h;

		int areaIndex = 0;

		for (int x = 0; x < max_x; x += location_step) for (int y = 0; y < max_y; y += location_step)
		{
			// Capture the image region in the searching bounding box
			Mat db_region_img(db_img, Rect(x, y, scale_w, scale_h));
			// Apply the pixel-by-pixel comparison method
			double tempScore = compareImgs(db_region_img, src_input);

			areaScore[areaIndex++] = tempScore;

			// Store the top k(k=score_size) match bounding box and score
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

		// Visualization
		if (visualize) {

			double max_difference = *max_element(areaScore, areaScore + areaIndex);

			int visIndex = 0;

			for (int x = 0; x < max_x; x += location_step) for (int y = 0; y < max_y; y += location_step)
			{
				Mat area = visImg(Rect(x, y, scale_w, scale_h));
				Mat color(area.size(), CV_8UC3, Scalar(0, 255, 0));
				double alphaValue = (1 - areaScore[visIndex++] / max_difference) * 0.7;
				addWeighted(color, alphaValue, area, 1.0 - alphaValue, 0.0, area);
				rectangle(visImg, Point(x, y), Point(x + scale_w, y + scale_h), Scalar(255, 255, 0));
			}

			imshow("Visualization", visImg);
			cvWaitKey(0);
			db_img.copyTo(visImg);

		}

	}

	// Draw the best match[top k (k=score_size)] rectangele
	for (int k = 0; k<score_size; k++) {
		Point start = Point(start_x[k], start_y[k]);
		Point end = Point(end_x[k], end_y[k]);
		rectangle(db_img, start, end, Scalar(255, 0, 0));
	}

	// You should keep this evalation code unchanged: 
	// Compare your detection boulding box with ground truth bouding box by IoU 
	// First we define the location of ground truth bouding box
	const int gt_start_x[10] = { 266,220,200,238,350,26,204,128,33,380 };
	const int gt_start_y[10] = { 146,248,83,120,80,10,347,258,196,207 };
	const int gt_end_x[10] = { 353,380,324,314,391,78,248,156,75,404 };
	const int gt_end_y[10] = { 233,398,207,196,121,62,391,288,238,231 };
	// Draw ground truth bouding box
	Point start = Point(gt_start_x[DETECTION_IMAGE - 1], gt_start_y[DETECTION_IMAGE - 1]);
	Point end = Point(gt_end_x[DETECTION_IMAGE - 1], gt_end_y[DETECTION_IMAGE - 1]);
	rectangle(db_img, start, end, Scalar(0, 0, 255));
	int gt_area = (gt_end_x[DETECTION_IMAGE - 1] - gt_start_x[DETECTION_IMAGE - 1]) * (gt_end_y[DETECTION_IMAGE - 1] - gt_start_y[DETECTION_IMAGE - 1]);
	// Calculate top 10 IoU, and print the best one
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

	// Show and store the detection reuslts
	imshow("Best Match Image", db_img);
	sprintf_s(tempname, filename_len, "%s\\detectionResults\\%d.jpg", IMAGE_folder, DETECTION_IMAGE);
	imwrite(tempname, db_img);

	printf("Done, Time elapsed = %.3f seconds \n", (std::clock() - startTime) / (double)CLOCKS_PER_SEC);

	if (visualize)
		printf("Visualization is enabled, thus time is not accurate for evaluation.\n");

	// Wait for the user to press a key in the GUI window.
	// Press ESC to quit
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
