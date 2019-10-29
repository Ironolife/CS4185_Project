/**
* CS4185/CS5185 Multimedia Technologies and Applications
* Course Assignment
* Image Retrieval Task
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace cv;

#define IMAGE_folder "D:\\dataset" // change to your folder location
#define IMAGE_LIST_FILE "dataset1" //the dataset1 for retrieval
#define output_LIST_FILE "searchResults" //the search results will store in this file
#define SEARCH_IMAGE "999.jpg" //change from 990 to 999 as the search images to get your output
#define INDEX_extension ".yml" // file type for indexing

struct features {
	bool hasPentagons;
	bool hasCircles;
	Mat histogram;
};

features searchFeatures;

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

	/*Mat drawing = Mat::zeros(gray.size(), CV_8UC3);

	for (int i = 0; i < contours_poly.size(); i++) {
		drawContours(drawing, hull, i, Scalar(255, 0, 0), 1, 8, vector<Vec4i>(), 0, Point());
	}

	namedWindow("Pentagons", CV_WINDOW_AUTOSIZE);
	imshow("Pentagons", drawing);

	waitKey(0);*/

	for (int i = 0; i < hull.size(); i++) {
		if (hull[i].size() == 5)
			return true;
	}

	return false;
}

bool findCircles(Mat img)
{
	// Convert to grayscale
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Apply blur
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);
	medianBlur(gray, gray, 5);

	// Find circles
	vector<Vec3f> circles;
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows, 60, 75, 50, 0);

	/*Mat drawing = Mat::zeros(gray.size(), CV_8UC3);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(drawing, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(drawing, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	namedWindow("Circles", CV_WINDOW_AUTOSIZE);
	imshow("Circles", drawing);

	waitKey(0);*/

	return circles.size() > 0;
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

features getFeatures(Mat img, int index)
{
	features features;

	// Set index file name
	const int filename_len = 900;
	char indexName[filename_len];
	sprintf_s(indexName, filename_len, "%s\\%s\\%s%s", IMAGE_folder, IMAGE_LIST_FILE, std::to_string(index), INDEX_extension);

	FileStorage fr(indexName, FileStorage::READ);

	if (fr["pentagons"].size() > 0) { // File exists

		fr["pentagons"] >> features.hasPentagons;
		fr["circles"] >> features.hasCircles;

		if (!features.hasPentagons || !features.hasCircles) {
			fr.release();
			return features;
		}

		fr["histogram"] >> features.histogram;
		fr.release();
		return features;

	}
	else { // File not exists

		features.hasPentagons = findPentagons(img);
		features.hasCircles = findCircles(img);

		FileStorage fw(indexName, FileStorage::WRITE);
		fw << "pentagons" << features.hasPentagons;
		fw << "circles" << features.hasCircles;

		if (!features.hasPentagons || !features.hasCircles) {
			fw.release();
			return features;
		}

		features.histogram = getHistogram(img);
		fw << "histogram" << features.histogram;
		fw.release();
		return features;

	}
}

// Compute similarity
double compareImgs(Mat db_img, int db_index)
{
	if (searchFeatures.histogram.empty())
		return DBL_MAX;

	features db_img_features = getFeatures(db_img, db_index);

	if (!db_img_features.hasPentagons || !db_img_features.hasCircles)
		return DBL_MAX;

	return compareHist(searchFeatures.histogram, db_img_features.histogram, 1);
}

int main(int argc, char** argv)
{
	std::clock_t start;
	start = std::clock();

	Mat src_input;
	Mat db_img;

	const int filename_len = 900;
	char tempname[filename_len];

	const int db_size = 1000;
	int db_id = 0;

	const int score_size = 10; // Change this to control return top n images
	double minscore[score_size] = { DBL_MAX };
	int minFilename[score_size];

	char minimg_name[filename_len];
	Mat min_img;

	sprintf_s(tempname, filename_len, "%s\\%s\\%s", IMAGE_folder, IMAGE_LIST_FILE, SEARCH_IMAGE);
	src_input = imread(tempname); // Read input image
	if (!src_input.data)
	{
		printf("Cannot find the input image!\n");
		system("pause");
		return -1;
	}
	imshow("Input", src_input);

	int searchIndex = atoi(((string)SEARCH_IMAGE).erase(((string)SEARCH_IMAGE).find(".jpg"), 4).c_str());

	searchFeatures = getFeatures(src_input, searchIndex);

	//Read Database
	for (db_id; db_id<db_size; db_id++) {
		sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, db_id);
		db_img = imread(tempname); // Read database image
		if (!db_img.data)
		{
			printf("Cannot find the database image number %d!\n", db_id + 1);
			system("pause");
			return -1;
		}

		// Apply the pixel-by-pixel comparison method
		double tempScore = compareImgs(db_img, db_id);

		printf("%s done!\n", tempname);

		// Store the top k min score ascending
		for (int k = 0; k<score_size; k++) {
			if (tempScore < minscore[k]) {
				for (int k1 = score_size - 1; k1>k; k1--) {
					minscore[k1] = minscore[k1 - 1];
					minFilename[k1] = minFilename[k1 - 1];
				}
				minscore[k] = tempScore;
				minFilename[k] = db_id;
				break;
			}
		}
	}

	printf("\n");

	// Read the top k max score image and write them to the a designated folder
	for (int k = 0; k<score_size; k++) {
		sprintf_s(minimg_name, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, minFilename[k]);
		min_img = imread(minimg_name);
		printf("the most similar image %d is %d.jpg, the histogram chi-squared difference is %.3f\n", k + 1, minFilename[k], minscore[k]);
		sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, output_LIST_FILE, minFilename[k]);
		imwrite(tempname, min_img);
	}

	// Output your precesion and recall (the ground truth are from 990 to 999)
	int count = 0;
	for (int k = 0; k<score_size; k++) {
		if (minFilename[k] >= 990 && minFilename[k] <= 999) {
			count++;
		}
	}
	double precision = (double)count / score_size;
	double recall = (double)count / 10;

	printf("\nthe precision and the recall for %s is %.2f and %.2f.\n", SEARCH_IMAGE, precision, recall);

	printf("Done, Time elapsed = %.3f seconds \n", (std::clock() - start) / (double)CLOCKS_PER_SEC);

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
