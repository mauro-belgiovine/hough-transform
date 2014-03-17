hough-transform
===============


Various implementation for Hough Transform (all OpenCV functions are referred to 2.4.8):

1) main.cpp + hough.cpp + hough.h: Hough Transform C++ plain code, with some OpenCV utilities (from http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html)

2) houghlines.cpp: show OpenCV HoughLinesP (probabilistic) execution time on CPU and GPU

3) hough_partialcircle.cpp: show cirlces and partial-circles tracked in an input image

4) HoughLines_Demo.cpp: edited OpenCV HoughLines_Demo source to generate input image starting from a list of polar coordinates (atlas_track_GPU/random1.txt) [previous version in hough_tracks.cpp]

5) hough_circle.cpp: plain C/C++ implementation for Hough Circle Transform (unfixed radius)

