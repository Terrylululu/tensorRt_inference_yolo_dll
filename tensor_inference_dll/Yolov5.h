#pragma once
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "cuda_runtime_api.h"
#include "NvInferRuntimeCommon.h"
#include "logging.h"

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>


using namespace std;
using namespace nvinfer1;
class TimerC
{
public:
	TimerC() : beg_(std::chrono::system_clock::now()) {}
	void reset() { beg_ = std::chrono::system_clock::now(); }

	void out(std::string message = "") {
		auto end = std::chrono::system_clock::now();
		std::cout << message << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg_).count() << "ms" << std::endl;
		reset();
	}
private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	chrono::time_point<std::chrono::system_clock> beg_;
};





#define DEVICE 0

namespace Yolo
{
	static constexpr int LOCATIONS = 4;
	static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;

	struct alignas(float) Detection {
		//center_x center_y w h
		float bbox[LOCATIONS];
		float conf;  // bbox_conf * cls_conf
		float class_id[2];
	};

	struct DetectionBox
	{
		cv::Rect box;
		float conf{};
		int classId{};
	};
}

class Yolov5Inter
{

public:
	int INPUT_H;
	int INPUT_W;
	int Batchsize=1;
	int CLASSES;
	vector<string> class_names;
	double CONF_THRESH = 0.65;
	double NMS_THRESH = 0.1;
	const char* INPUT_BLOB_NAME = "images";
	const char* OUTPUT_BLOB_NAME = "output0";

	void* buffers[2];
	int OUTPUT_SIZE;
	int inputIndex;
	int outputIndex;

	IRuntime* mRuntime;
	ICudaEngine* mEngine;
	IExecutionContext* context;
	cudaStream_t stream;

	Yolov5Inter();

	Yolov5Inter(string path,string name_path, int width, int height, int classNum);

	~Yolov5Inter();

	std::vector<Yolo::DetectionBox> inference(cv::Mat img, cv::Mat& dstImg);
	
	void doInference(IExecutionContext& context, float* input, float* output, int batchSize);

	void initContext(string path);
	void preprocess_img(cv::Mat& img, int input_w, int input_h, float* data);


};

