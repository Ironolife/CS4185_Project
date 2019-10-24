/**
* CS4185/CS5185 Multimedia Technologies and Applications
* Course Assignment
* Image Retrieval Task
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <iostream>
#include <stdio.h>
#include <algorithm>

using namespace std;
using namespace cv;

#define IMAGE_folder "H:\\dataset" // change to your folder location
#define IMAGE_LIST_FILE "dataset1" //the dataset1 for retrieval
#define output_LIST_FILE "searchResults" //the search results will store in this file
#define SEARCH_IMAGE "999.jpg" //change from 990 to 999 as the search images to get your output

struct features {
	Mat histograms[5];
};

features getHist(Mat img)
{
	//Convert to HSV
	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);

	//Find image dimensions & centers
	int width = hsv.size().width, height = hsv.size().height;
	int centerX = width / 2, centerY = height / 2;

	//Split image into 5 segments
	Mat segMasks[5];
	int segRectPoints[4][4] = {
		{ 0, centerX, 0, centerY },
		{centerX, width, 0, centerY},
		{ 0, centerX, centerY, height},
		{centerX, width, centerY, height},
	};

	Mat centerMask = Mat::zeros(hsv.size(), 0);
	ellipse(centerMask, Point(centerX, centerY), Point(centerX * 0.75, centerY * 0.75), 0, 0, 360, Scalar(255, 255, 255), -1);

	for (int i = 0; i < 4; i++) {
		segMasks[i] = Mat::zeros(hsv.size(), 0);
		rectangle(segMasks[i], Point(segRectPoints[i][0], segRectPoints[i][2]), Point(segRectPoints[i][1], segRectPoints[i][3]), Scalar(255, 255, 255), -1);
		subtract(segMasks[i], centerMask, segMasks[i]);
	}

	//Calculate histogram for each region
	int hBins = 8, sBins = 12;
	int histSize[] = { hBins, sBins };
	float hRanges[] = { 0, 180 }, sRanges[] = { 0, 256 };
	const float* ranges[] = { hRanges, sRanges };
	int channels[] = { 0, 1 };
	features features;
	for (int i = 0; i < 5; i++) {
		calcHist(&hsv, 1, channels, segMasks[i], features.histograms[i], 2, histSize, ranges);
		normalize(features.histograms[i], features.histograms[i], 0, 1, NORM_MINMAX, -1, Mat());
	}

	return features;
}

double featureMatching(Mat img1, Mat img2)
{
	// detecting keypoints
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	// computing descriptors
	SiftDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// computing keypoints distance range
	double maxDistance = 0;
	double minDistance = 100;
	for (int i = 0; i < descriptors1.rows; i++) {
		double distance = matches[i].distance;
		if (distance < minDistance) minDistance = distance;
		if (distance > maxDistance) maxDistance = distance;
	}

	// computing dissimilarity score
	vector<DMatch> goodMatches;
	double score = 0;
	for (int i = 0; i < descriptors1.rows; i++) {
		double distance = matches[i].distance;
		if (distance <= max(1.5 * minDistance, 0.02)) {
			goodMatches.push_back(matches[i]);
			score += distance;
		}
	}

	/*Mat mat_img;
	drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, mat_img, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Good Matches", mat_img);
	waitKey(0);*/

	if (goodMatches.size() == 0) {
		score = DBL_MAX;
	}

	return score;
}

int findCircle(Mat img)
{
	// convert to grayscale
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// apply blur
	medianBlur(gray, gray, 5);

	// find circles
	vector<Vec3f> circles;
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows/8, 60, 75, 50, 0);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(gray, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(gray, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	/*namedWindow("circle", CV_WINDOW_AUTOSIZE);
	imshow("circle", gray);
	waitKey(0);*/

	return circles.size();
}

//Compute similarity
double compareImgs(Mat img1, Mat img2)
{
	double score = 0;

	features features1 = getHist(img1);
	features features2 = getHist(img2);

	for (int i = 0; i < 5; i++) {
		score += compareHist(features1.histograms[i], features2.histograms[i], 1);
	}

	score *= featureMatching(img1, img2);

	return score;
}

int main(int argc, char** argv)
{
	Mat src_input;
	Mat db_img;

	const int filename_len = 900;
	char tempname[filename_len];

	const int db_size = 1000;
	int db_id = 0;

	const int score_size = 10; //Change this to control return top n images
	double minscore[score_size] = { DBL_MAX };
	int minFilename[score_size];

	char minimg_name[filename_len];
	Mat min_img;

	sprintf_s(tempname, filename_len, "%s\\%s\\%s", IMAGE_folder, IMAGE_LIST_FILE, SEARCH_IMAGE);
	src_input = imread(tempname); //Read input image
	if (!src_input.data)
	{
		printf("Cannot find the input image!\n");
		system("pause");
		return -1;
	}
	imshow("Input", src_input);

	//Read Database
	for (db_id; db_id<db_size; db_id++) {
		sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, db_id);
		db_img = imread(tempname); //Read database image
		if (!db_img.data)
		{
			printf("Cannot find the database image number %d!\n", db_id + 1);
			system("pause");
			return -1;
		}

		//Apply the pixel-by-pixel comparison method
		double tempScore = compareImgs(src_input, db_img);

		printf("%s done!\n", tempname);

		//Store the top k min score ascending
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

	//Read the top k max score image and write them to the a designated folder
	for (int k = 0; k<score_size; k++) {
		sprintf_s(minimg_name, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, minFilename[k]);
		min_img = imread(minimg_name);
		printf("the most similar image %d is %d.jpg, the pixel-by-pixel difference is %f\n", k + 1, minFilename[k], minscore[k]);
		sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, output_LIST_FILE, minFilename[k]);
		imwrite(tempname, min_img);
		//imshow(tempname,max_img);
	}

	//Output your precesion and recall (the ground truth are from 990 to 999)
	int count = 0;
	for (int k = 0; k<score_size; k++) {
		if (minFilename[k] >= 990 && minFilename[k] <= 999) {
			count++;
		}
	}
	double precision = (double)count / score_size;
	double recall = (double)count / 10;

	printf("the precision and the recall for %s is %.2f and %.2f.\n", SEARCH_IMAGE, precision, recall);

	printf("Done \n");

	//Wait for the user to press a key in the GUI window.
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
