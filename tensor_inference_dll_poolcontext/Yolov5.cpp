
#include "Yolov5.h"
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
void getBestClassInfo(float* it, const int& numClasses,
	float& bestConf, int& bestClassId)
{
	// first 5 element are box and obj confidence
	bestClassId = 5;
	bestConf = 0;

	for (int i = 5; i < numClasses + 5; i++)
	{
		if (it[i] > bestConf)
		{
			bestConf = it[i];
			bestClassId = i - 5;
		}
	}

}

void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
	float gain = min((float)imageShape.height / (float)imageOriginalShape.height,
		(float)imageShape.width / (float)imageOriginalShape.width);

	int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
				  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

	coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
	coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

	coords.width = (int)std::round(((float)coords.width / gain));
	coords.height = (int)std::round(((float)coords.height / gain));

}

std::vector<Yolo::DetectionBox> nms(float *it, float conf_thresh, int width, int height,int outsize,int numClasses, float nms_thresh = 0.5) {
	//int det_size = sizeof(Yolo::Detection) / sizeof(float);
	std::map<float, std::vector<Yolo::Detection>> m;
	std::vector<cv::Rect> boxes;
	std::vector<float> confs;
	std::vector<int> classIds;

	int step = numClasses + 5;
	for (int i = 0; i < outsize; i += step) {

		float* pp = it + i;
		float clsConf = pp[4];

		if (clsConf > conf_thresh)
		{
			int centerX = (int)(pp[0]);
			int centerY = (int)(pp[1]);
			int width = (int)(pp[2]);
			int height = (int)(pp[3]);
			int left = centerX - width / 2;
			int top = centerY - height / 2;

			float objConf;
			int classId;
			getBestClassInfo(pp, numClasses, objConf, classId);//找到属于哪一类

			float confidence = clsConf * objConf;

			boxes.emplace_back(left, top, width, height);
			confs.emplace_back(confidence);
			classIds.emplace_back(classId);
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, conf_thresh, nms_thresh, indices);
	// std::cout << "amount of NMS indices: " << indices.size() << std::endl;

	std::vector<Yolo::DetectionBox> detections;

	for (int idx : indices)
	{
		Yolo::DetectionBox det;
		det.box = cv::Rect(boxes[idx]);
		//调整比例600, 600
		scaleCoords(cv::Size(640, 640), det.box, cv::Size(width, height));//计算无灰色填充时，相对于640，640的坐标

		det.conf = confs[idx];
		det.classId = classIds[idx];
		detections.emplace_back(det);
	}

	return detections;

}


void Yolov5Inter::preprocess_img(cv::Mat& img, int input_w, int input_h,float *data) {
	int w, h, x, y;
	float r_w = input_w / (img.cols * 1.0);
	float r_h = input_h / (img.rows * 1.0);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));


	int i = 0;
	for (int row = 0; row < input_h; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < input_w; ++col) {
			//bgr格式数据 归一化
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + input_h * input_w] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * input_h * input_w] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

}

Yolov5Inter::Yolov5Inter()
{
}

Yolov5Inter::Yolov5Inter(string path, string name_path,int width, int height, int classNum)
{
	this->INPUT_H = height;
	this->INPUT_W = width;
	this->CLASSES = classNum;

	string classesFile = name_path;
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	assert(this->CLASSES == class_names.size());
	


	//this->OUTPUT_SIZE = 22743 * (5+ classNum) * sizeof(float);
	int size = (pow((width / 32),2) + pow(( width / 16),2) + pow((width / 8),2)) * 3;
	this->OUTPUT_SIZE = size * (5 + classNum) /* sizeof(float)*/;
	
	this->initContext(path);
	//assert(mEngine->getNbBindings() == 2);8.5版本弃用
	assert(mEngine->getNbIOTensors() == 2);//返回输入输出张量的总数
	inputIndex = mEngine->getBindingIndex(INPUT_BLOB_NAME);//8.5版本弃用
	outputIndex = mEngine->getBindingIndex(OUTPUT_BLOB_NAME);

	// Create GPU buffers on device 
	CHECK(cudaMalloc(&buffers[inputIndex], this->Batchsize * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], this->Batchsize * OUTPUT_SIZE * sizeof(float)));
	//Create GPU stream on device
	CHECK(cudaStreamCreate(&stream));
}

Yolov5Inter::~Yolov5Inter()
{
	context->destroy();
	mEngine->destroy();
	mRuntime->destroy();

	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	// Release stream and buffers
	CHECK(cudaStreamDestroy(stream));
}

std::vector<Yolo::DetectionBox> Yolov5Inter::inference(cv::Mat img, cv::Mat & dstImg)
{
	TimerC time;
	float* data = new float[3 * INPUT_H * INPUT_W];
	float *prob = new float[OUTPUT_SIZE];
	time.out("host malloc cost:");

	int rw = img.cols;
	int rh = img.rows;


	preprocess_img(img, INPUT_W, INPUT_H, data); // 1. BGR to RGB 2.resize 3.img.data to data
	

	time.out("yolo pre cost:");
	//执行推理

	auto context = pool->acquireContext();

	doInference(*context, data, prob, this->Batchsize);
	// 当context离开作用域，智能指针会自动释放，
	 // 上下文回到池中，但其他线程不知道，可以调用通知
	
	pool->notifyContextReleased(); //存在的风险，当前函数没有执行完，局部变量context没有被释放，就通知下一个线程

	time.out("yolo inter cost:");

	std::vector<Yolo::DetectionBox> detResult = nms(prob, CONF_THRESH,rw,rh,OUTPUT_SIZE,CLASSES, NMS_THRESH);


	for (size_t j = 0; j < detResult.size(); j++) {
		cv::Rect r = detResult[j].box;
		cv::rectangle(dstImg, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
		
		cv::putText(dstImg, this->class_names[(int)detResult[j].classId] + std::to_string((int)detResult[j].classId) + "conf:" + std::to_string(detResult[j].conf).substr(0,4), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0xFF), 2);
	}

	delete[] data;
	delete[] prob;
	time.out("yolo post cost:");
	return detResult;
}

void Yolov5Inter::doInference(IExecutionContext & context, float * input, float * output, int batchSize)
{
	TimerC time;
	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

	//context.setBindingDimensions(inputIndex, Dims4(batchSize, 3, INPUT_H, INPUT_W));
	context.enqueueV2(buffers, stream, nullptr);

	time.out("cuda enq time:");

	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	//流同步：通过cudaStreamSynchronize()来协调。
	
	time.out("cuda sync time:");

}

void Yolov5Inter::initContext(string path)
{
	cudaError_t err =cudaSetDevice(DEVICE);
	
	if (err != cudaSuccess) {
		printf("无法设置设备: %s\n", cudaGetErrorString(err));
	}
	Logger gLogger;
	//static Logger gLogger;
	this->mRuntime = createInferRuntime(gLogger);

	assert(mRuntime != nullptr);

	size_t size{ 0 };
	char *trtModelStream{ nullptr };
	std::ifstream file(path, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}
	this->mEngine = mRuntime->deserializeCudaEngine(trtModelStream, size);

	// 2. 创建上下文池
	this->pool = new TensorRTContextPool(mEngine, 3);

	//this->context = mEngine->createExecutionContext();

	delete[] trtModelStream;
}
