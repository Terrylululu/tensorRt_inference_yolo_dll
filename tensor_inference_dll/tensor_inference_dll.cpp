
#include "dirent.h"
#include "Yolov5.h"
#include "NvOnnxParser.h"
#include <thread>


using namespace std;
using namespace cv;
int read_files_in_dir2(const char* p_dir_name, std::vector<std::string>& file_names) {
	DIR* p_dir = opendir(p_dir_name);
	if (p_dir == nullptr) {
		return -1;
	}

	struct dirent* p_file = nullptr;
	while ((p_file = readdir(p_dir)) != nullptr) {
		if (strcmp(p_file->d_name, ".") != 0 &&
			strcmp(p_file->d_name, "..") != 0) {

			// 只寻找 jpg, bmp, png 格式的文件
			std::string file_name(p_file->d_name);
			std::string extension = file_name.substr(file_name.find_last_of('.') + 1);
			if (extension == "jpg" || extension == "jpeg" || extension == "bmp" || extension == "png") {
				/*std::string cur_file_name(p_dir_name);
				cur_file_name += "/";
				cur_file_name += file_name;*/
				std::string cur_file_name(p_file->d_name);
				file_names.push_back(cur_file_name);
			}
		}
	}

	closedir(p_dir);
	return 0;
}

void gen_train_voc2007_img()
{
	Yolov5Inter* yolo;
	yolo = new Yolov5Inter("VOC2007/weights/best.trt","VOC2007/images/voc.names" ,640, 640, 20);//模型20类，训练尺寸为640，640
	yolo->CONF_THRESH = 0.3;//随便训练的，阈值设置低一点
	std::vector<string> files;
	string rootPath = "VOC2007/images/";
	string dstPath = "VOC2007/OutputImage/";
	read_files_in_dir2(rootPath.c_str(), files);

	int j = 0;
	for (string file : files)
	{
		std::cout << file << std::endl;
		Mat img = cv::imread(rootPath + file);
		Mat colorImg = img.clone();
		TimerC time;
		std::vector<Yolo::DetectionBox> boxs = yolo->inference(img, colorImg);
		time.out("yolo:");
		int k = 0;
		if (size(boxs) != 0)
		{
			imwrite(dstPath + file, colorImg);
		}

		j++;
	}

}

void gen_train_luowen_img()
{
	Yolov5Inter* yolo;
	yolo = new Yolov5Inter("LuoWen/weights/best.trt","LuoWen/images/voc.names" ,640, 640, 6);//模型6类，训练尺寸为640，640
	std::vector<string> files;
	string rootPath = "LuoWen/images/";

	string dstPath = "LuoWen/OutputImage/";

	read_files_in_dir2(rootPath.c_str(), files);

	int j = 0;
	for (string file : files)
	{
		std::cout << file << std::endl;
		Mat img = cv::imread(rootPath + file);
		Mat colorImg = img.clone();
		TimerC time;
		std::vector<Yolo::DetectionBox> boxs = yolo->inference(img, colorImg);
		time.out("yolo:");
		int k = 0;

		imwrite(dstPath + file, colorImg);

		j++;
	}

}

int  main()
{

	//gen_train_voc2007_img();
	gen_train_luowen_img();
	
	return 1;
}
