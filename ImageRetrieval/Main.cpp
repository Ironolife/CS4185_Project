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

using namespace std;
using namespace cv;

#define IMAGE_folder "D:\\dataset"     // change to your folder location
#define IMAGE_LIST_FILE "dataset1"         //the dataset1 for retrieval
#define output_LIST_FILE "searchResults"  //the search results will store in this file
#define SEARCH_IMAGE "999.jpg"   //change from 990 to 999 as the search images to get your output

/**
* @function main
*/

//Compute pixel-by-pixel difference
double compareImgs(Mat img1, Mat img2)
{
	int w = img1.cols, h = img1.rows;
	Mat new_img2;
	resize(img2, new_img2, img1.size());
	double sum = 0;
	for (int i = 0; i < w; i++)for (int j = 0; j < h; j++)
	{
		sum += abs(img1.at<uchar>(j, i) - new_img2.at<uchar>(j, i));
	}
	return sum;
}

int main(int argc, char** argv)
{
	Mat src_input, gray_input;
	Mat db_img, db_gray_img;

	const int filename_len = 900;
	char tempname[filename_len];

	const int db_size = 1000;
	int db_id = 980;

	const int score_size = 10;   //change this to control return top n images
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
	cvtColor(src_input, gray_input, COLOR_BGR2GRAY); // Convert to grayscale

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
		cvtColor(db_img, db_gray_img, COLOR_BGR2GRAY); // Convert to grayscale

													   // Apply the pixel-by-pixel comparison method
		double tempScore = compareImgs(gray_input, db_gray_img);

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
