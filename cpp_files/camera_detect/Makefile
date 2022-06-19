CXX = g++
CXXFLAGS = -O3 -Wall -c -fmessage-length=0

CAMERA_DETECT =	camera_detect

INC_DIRS = -I . -I /usr/local/include/opencv4
LIB_DIRS = -L . -L /usr/local/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_ml -lopencv_flann -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_videoio -lopencv_dnn

all: $(CAMERA_DETECT)

SRC_FILES = $(wildcard *.cpp)
CAMERA_DETECT_OBJ = $(patsubst %.cpp,%.o,$(SRC_FILES))

$(CAMERA_DETECT_OBJ): $(SRC_FILES)
	$(CXX) -std=c++17 -g $(INC_DIRS) -c $(CXXFLAGS) $^

$(CAMERA_DETECT): $(CAMERA_DETECT_OBJ)
	$(CXX) -std=c++17 $(LIB_DIRS) -o $@ $^ $(OPENCV_LIBS)

clean:
	rm -f $(CAMERA_DETECT_OBJ) $(CAMERA_DETECT)
