# (C)2013, Bruno Keymolen
# http://www.keymolen.com
# http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html
CXX=g++
CC=gcc
NVCC=nvcc
OPTFLAGS=-g3 -ggdb -O0 -std=c++11 -m64
CXXFLAGS=-Wall -I. -I/usr/local/include $(OPTFLAGS) -I/usr/local/cuda-5.5/include -I/home/belgiovi/NVIDIA_CUDA-5.0_Samples/common/inc
CFLAGS=-Wall $(OPTFLAGS)
NVCCFLAGS=-I. -I/usr/local/include -I/usr/local/cuda-5.5/include -I/home/belgiovi/NVIDIA_CUDA-5.0_Samples/common/inc -gencode arch=compute_20,code=sm_21
LDFLAGS= -L/usr/local/lib 

LDFLAGS+= -lopencv_highgui -lopencv_core -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_objdetect -lopencv_ts -lopencv_gpu

SRC = 	main.o hough.o
SRC_DEMO = HoughLines_Demo.o
SRC_TRANSFORM = houghlines.o
SRC_CIRCLE = hough_partialcircle.o
SRC_TRACKS = hough_tracks.o
SRC_SOURCE_CIRCLE = hough_circle.o
	
all: hough houghlines hough_partialcircle hough_tracks HoughLines_Demo hough_circle

hough: $(SRC) $(MODULES)
	$(NVCC) $(LDFLAGS) $(MODULES) $(SRC) -o hough

houghlines: $(SRC_TRANSFORM) $(MODULES)
	$(CXX) $(LDFLAGS) $(OPTFLAGS) $(MODULES) $(SRC_TRANSFORM) -o houghlines
	
hough_partialcircle: $(SRC_CIRCLE)
	$(CXX) $(LDFLAGS) $(OPTFLAGS) $(SRC_CIRCLE) -o hough_partialcircle
	
hough_tracks: $(SRC_TRACKS)
	$(CXX) $(LDFLAGS) $(OPTFLAGS) $(SRC_TRACKS) -o hough_tracks
	
HoughLines_Demo: $(SRC_DEMO)
	$(CXX) $(LDFLAGS) $(OPTFLAGS) $(SRC_DEMO) -o HoughLines_Demo
	
hough_circle: $(SRC_SOURCE_CIRCLE)
	$(CXX) $(LDFLAGS) $(OPTFLAGS) $(SRC_SOURCE_CIRCLE) -o hough_circle
	
# doc_example: $(SRC_EXAMPLE) $(MODULES)
#	$(CXX) $(LDFLAGS) $(MODULES) $(SRC_EXAMPLE) -o doc_example
	
# doc_transform: $(SRC_TRANSFORM) $(MODULES)
#	$(CXX) $(LDFLAGS) $(MODULES) $(SRC_EXAMPLE) -o doc_transform

%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f *.o hough doc_example doc_transform houghlines hough_partialcircle hough_tracks hough_circle HoughLines_Demo

PREFIX ?= /usr

install: all
	install -d $(PREFIX)/bin
	install hough  $(PREFIX)/bin

.PHONY: clean all hough install
