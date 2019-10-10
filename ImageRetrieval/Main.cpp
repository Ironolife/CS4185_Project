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

double featureMatching(Mat img1, Mat img2)
{
	// convert to grayscale
	Mat gray1, gray2;
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);

	// detecting keypoints
	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(gray1, keypoints1);
	detector.detect(gray2, keypoints2);

	// computing descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(gray1, keypoints1, descriptors1);
	extractor.compute(gray2, keypoints2, descriptors2);

	// matching descriptors
	BruteForceMatcher<L2<float> > matcher;
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
		if (distance <= max(2 * minDistance, 0.02)) {
			goodMatches.push_back(matches[i]);
			score += distance;
		}
	}

	return score;
}

//Compute similarity
double compareImgs(Mat img1, Mat img2)
{
	double score = 0;

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

	const int score_size = 10; //change this to control return top n images
	double minscore[score_size] = { DBL_MAX };
	int minFilename[score_size];

	char minimg_name[filename_len];
	Mat min_img;

	sprintf_s(tempname, filename_len, "%s\\%s\\%s", IMAGE_folder, IMAGE_LIST_FILE, SEARCH_IMAGE);
	src_input = imread(tempname); // read input image
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
		db_img = imread(tempname); // read database image
		if (!db_img.data)
		{
			printf("Cannot find the database image number %d!\n", db_id + 1);
			system("pause");
			return -1;
		}

		// Apply the pixel-by-pixel comparison method
		double tempScore = compareImgs(src_input, db_img);

		//store the top k min score ascending
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

	//read the top k max score image and write them to the a designated folder
	for (int k = 0; k<score_size; k++) {
		sprintf_s(minimg_name, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, IMAGE_LIST_FILE, minFilename[k]);
		min_img = imread(minimg_name);
		printf("the most similar image %d is %d.jpg, the pixel-by-pixel difference is %f\n", k + 1, minFilename[k], minscore[k]);
		sprintf_s(tempname, filename_len, "%s\\%s\\%d.jpg", IMAGE_folder, output_LIST_FILE, minFilename[k]);
		imwrite(tempname, min_img);
		//imshow(tempname,max_img);
	}

	//output your precesion and recall (the ground truth are from 990 to 999)
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
