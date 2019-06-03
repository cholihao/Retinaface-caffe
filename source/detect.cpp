#include "anchor_generator.h"
//#include "inference.hpp"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "detect.h"
#include "tools.h"

using namespace std;

void printMat(const cv::Mat &image)
{
    uint8_t  *myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int _stride = image.step;//in case cols != strides
    for(int i = 0; i < height; i++)
    {
	for(int j = 0; j < width; j++)
	{
	    uint8_t  val = myData[ i * _stride + j];
	    cout << val;

	    //do whatever you want with your value
	}
    }
    cout << endl;
}

void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
    cv::Mat sample, resized, sample_float;
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    cv::resize(sample, resized, cv::Size(640,640));
    resized.convertTo(sample_float, CV_32FC3);
    //cout << "xTrainData (python)  = " << endl << format(resized, cv::Formatter::FMT_PYTHON) << endl << endl;
    cv::split(sample_float, *input_channels);
}

void wrapInputLayer(std::shared_ptr<caffe::Net<float> > net_, std::vector<cv::Mat>* input_channels)
{
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
	cv::Mat channel(height, width, CV_32FC1, input_data);
	input_channels->push_back(channel);
	input_data += width * height;
    }
}

std::vector<Anchor>  detect_face(cv::Mat& input)
{
    assert(input.rows == fd_h_ && input.cols == fd_w_ && input.channels() == fd_c_);
    std::vector<cv::Mat> fd_input_channels;
    wrapInputLayer(fd_net_, &fd_input_channels);
    preprocess(input, &fd_input_channels);

    fd_net_->Forward();

    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
	int stride = _feat_stride_fpn[i];
	ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i)
    {
	char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
	char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
	char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
	const caffe::Blob<float>* clsBlob = fd_net_->blob_by_name(clsname).get();
	const caffe::Blob<float>* regBlob = fd_net_->blob_by_name(regname).get();
	const caffe::Blob<float>* ptsBlob = fd_net_->blob_by_name(ptsname).get();
	ac[i].FilterAnchor(clsBlob, regBlob, ptsBlob, proposals);
	printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());
	for (int r = 0; r < proposals.size(); ++r) {
	    proposals[r].print();
	}

    }
    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    //printf("final result %d\n", result.size());
    //result[0].print();
    return result;
}

int main(int argc, char** argv) {

    //cv::Mat img = cv::imread("test.jpg");
    //cv::cvtColor(img, img, CV_BGR2RGB);

    // please replace your own inference code

    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    //caffe::Caffe::SetDevice(0);
    path_to_fd_model_ = std::string("/work/code/RetinaFace-Cpp/caffemodel/mnet.prototxt");
    string path_to_fd_caffemodel = path_to_fd_model_.substr(0 ,path_to_fd_model_.size()-8) + "caffemodel";
    fd_net_.reset(new caffe::Net<float>(path_to_fd_model_, caffe::TEST));
    fd_net_->CopyTrainedLayersFrom(path_to_fd_caffemodel);
    caffe::Blob<float>* input_layer = fd_net_->input_blobs()[0];
    fd_n_ = input_layer->shape(0);
    fd_c_ = input_layer->shape(1);
    fd_h_ = input_layer->shape(2);
    fd_w_ = input_layer->shape(3);
    input_layer->Reshape(1, fd_c_,fd_h_,fd_w_);
    fd_net_->Reshape();
    fd_n_ = 1;

    cv::Mat img = cv::imread(argv[1]);
    cv::Mat input = img.clone();

    std::vector<Anchor> result = detect_face(input);
    for(int i = 0; i < result.size(); i ++) {
	cv::rectangle(img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
    }
    cv::imwrite("result.jpg", img);

    return 0;
}
