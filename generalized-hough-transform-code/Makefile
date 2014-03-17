CXX=g++
CXXFLAGS=-Wall
LDFLAGS= -L/usr/local/lib -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_ts

SRC=GHT.o

all: gen-hough

gen-hough: $(SRC)
	$(CXX) $(LDFLAGS) $(SRC) -o gen-hough

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
        
clean:
	rm *.o gen-hough
