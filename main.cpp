// **********************************************************************************
//
// BSD License.
// This file is part of a Hough Transformation tutorial,
// see: http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html
//
// Copyright (c) 2013, Bruno Keymolen, email: bruno.keymolen@gmail.com
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or other
// materials provided with the distribution.
// Neither the name of "Bruno Keymolen" nor the names of its contributors may be
// used to endorse or promote products derived from this software without specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// **********************************************************************************

#include <dirent.h>
#include <cstring>
#include <map>
#include <iostream>
#include <cstdio>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "hough.h"

std::string img_path;
int threshold = 0;

//DEVICE code
#define CPU_DEV 0
#define GPU_DEV 1
int device = CPU_DEV;
std::string device_arg = "cpu";

const char* CW_IMG_ORIGINAL = "Result";
const char* CW_IMG_EDGE 	= "Canny Edge Detection";
const char* CW_ACCUMULATOR  = "Accumulator";

void doTransform(std::string, int threshold);


void usage(char * s)
{

	fprintf( stderr, "\n");
    fprintf( stderr, "%s -s <source file> [-t <threshold>] [-d <device>] - hough transform. build: %s-%s \n", s, __DATE__, __TIME__);
	fprintf( stderr, "   s: path image file\n");
	fprintf( stderr, "   t: hough threshold\n");
	fprintf( stderr, "   d: 'cpu' OR 'gpu'\n");
	fprintf( stderr, "\nexample: ./hough -s ./img/hangover-0232.jpg -t 80\n");
	fprintf( stderr, "\n");
}

int main(int argc, char** argv) {

	int c;
	
	
	while ( ((c = getopt( argc, argv, "d:s:t:?" ) ) ) != -1 )
	{
	    switch (c)
	    {
	    case 'd':
		device_arg = optarg;
		if(strcmp(device_arg.c_str(), "gpu") == 0){
		  device = GPU_DEV;
		}else if(strcmp(device_arg.c_str(), "cpu") == 0){
		  device = CPU_DEV;
		}else{
		  usage(argv[0]);
		  return -1;
		}
		break;
	    case 's':
	    	img_path = optarg;
	    	break;
	    case 't':
	    	threshold = atoi(optarg);
	    	break;
	    case '?':
	    default:
		usage(argv[0]);
		return -1;
	    }
	}

	if(img_path.empty())
	{
		usage(argv[0]);
		return -1;
	}

    cv::namedWindow(CW_IMG_ORIGINAL, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(CW_IMG_EDGE, 	 cv::WINDOW_AUTOSIZE);
    cv::namedWindow(CW_ACCUMULATOR,	 cv::WINDOW_AUTOSIZE);

    cvMoveWindow(CW_IMG_ORIGINAL, 10, 10);
    cvMoveWindow(CW_IMG_EDGE, 680, 10);
    cvMoveWindow(CW_ACCUMULATOR, 1350, 10);

    doTransform(img_path, threshold);

	return 0;
}



void doTransform(std::string file_path, int threshold)
{
	cv::Mat img_edge;
	cv::Mat img_blur;

	cv::Mat img_ori = cv::imread( file_path ,1 );
	cv::blur( img_ori, img_blur, cv::Size(5,5) );
	cv::Canny(img_blur, img_edge, 100, 150, 3);

	int w = img_edge.cols;
	int h = img_edge.rows;

	//Transform
	keymolen::Hough hough;
	
	const int64 start = cv::getTickCount();
	
	if(device == CPU_DEV){
	  hough.Transform(img_edge.data, w, h);
	}else if(device == GPU_DEV){
	  hough.Transform_GPU(img_edge.data, w, h);
	}
	
	const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << device_arg << " Time : " << timeSec * 1000 << " ms" << std::endl;


	if(threshold == 0)
		threshold = w>h?w/4:h/4;

	while(1)
	{
		cv::Mat img_res = img_ori.clone();

		//Search the accumulator
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines = hough.GetLines(threshold);

		//Draw the results
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;
		for(it=lines.begin();it!=lines.end();it++)
		{
			cv::line(img_res, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar( 0, 0, 255), 2, 8);
		}

		//Visualize all
		int aw, ah, maxa;
		aw = ah = maxa = 0;
		const unsigned int* accu = hough.GetAccu(&aw, &ah);

		for(int p=0;p<(ah*aw);p++)
		{
			if((int)accu[p] > maxa)
				maxa = accu[p];
		}
		double contrast = 1.0;
		double coef = 255.0 / (double)maxa * contrast;

		cv::Mat img_accu(ah, aw, CV_8UC3);
		for(int p=0;p<(ah*aw);p++)
		{
			unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
			img_accu.data[(p*3)+0] = 255;
			img_accu.data[(p*3)+1] = 255-c;
			img_accu.data[(p*3)+2] = 255-c;
		}


		cv::imshow(CW_IMG_ORIGINAL, img_res);
		cv::imshow(CW_IMG_EDGE, img_edge);
		cv::imshow(CW_ACCUMULATOR, img_accu);

		char c = cv::waitKey(360000);
		if(c == '+')
			threshold += 5;
		if(c == '-')
			threshold -= 5;
		if(c == 27)
			break;
	}
}

