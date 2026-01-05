#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/utility.hpp>
#include <omp.h>

int main() {
    omp_set_num_threads(omp_get_num_procs()); 

    
    cv::TickMeter tm;
    // std::cout << "omp_get_num_procs(): " << omp_get_num_procs() << "\n";
    // std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << "\n";

    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices available: " << deviceCount << std::endl;

    if (deviceCount <= 0) {
        std::cout << "No CUDA-capable device found or OpenCV built without CUDA." << std::endl;
        return 0;
    }


    std::string imgName = "data/frame-0.tif";
    cv::Mat img;
    cv::Mat imgBlur;
    cv::Size ksize(31, 31);
    double sigma = 11;
    int method = cv::TM_CCOEFF_NORMED;


    img = cv::imread(imgName, cv::IMREAD_UNCHANGED);
    cv::GaussianBlur(img, imgBlur, ksize, sigma);


    int h = img.rows;
    int w = img.cols;

    int cx = w/2;
    int cy = h/2;

    int ox = cx - cx/2;
    int oy = cy - cy/2;


    cv::Rect myROI(ox, oy, cx, cy);
    cv::Mat croppedImg = img(myROI);
    cv::Mat croppedImgBlur;
    cv::GaussianBlur(croppedImg, croppedImgBlur, ksize, sigma);

    cv::Mat cpuRes(h, w, CV_32F);
    double minVal_cpu, maxVal_cpu;
    cv::Point minLoc_cpu, maxLoc_cpu;

    // #pragma omp parallel 
    // #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < 420; i++ ){
        tm.start();
        cv::matchTemplate(imgBlur, croppedImgBlur, cpuRes, method);
        tm.stop();
    }

    std::cout << "CPU matching FPS: " << tm.getFPS()<< " FPS" << std::endl;
    tm.reset();


    cv::minMaxLoc(cpuRes, &minVal_cpu, &maxVal_cpu, &minLoc_cpu, &maxLoc_cpu);
    std::cout << "CPU results: confidence: " << maxVal_cpu << "; x: " << maxLoc_cpu.x << "; y: " << maxLoc_cpu.y << std::endl;



    std::cout << "Image size: " << w << "x" << h << std::endl;
    std::cout << "Cropped image size: " << croppedImg.cols << "x" << croppedImg.rows << std::endl;


    cv::imshow("Image", imgBlur);
    cv::imshow("Image", croppedImg);

    cv::waitKey(0);
    cv::destroyAllWindows();




    cv::cuda::DeviceInfo dev(0);
    std::cout << "Using device 0: " << dev.name() << std::endl;

    // // // Upload to GPU
    cv::cuda::GpuMat d_img, d_templ, d_res;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::Mat result;
    

    cv::Ptr<cv::cuda::TemplateMatching> matcher =
        cv::cuda::createTemplateMatching(CV_8UC1, method); 
   
    d_templ.upload(croppedImgBlur);
    
    for(int i = 0; i< 420; i++){
        tm.start();
        d_img.upload(imgBlur);
        matcher->match(d_img, d_templ, d_res);
        d_res.download(result);
        tm.stop();
    }

    std::cout << "GPU matching FPS: " << tm.getFPS() << " FPS" << std::endl;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "GPU results: confidence: " << maxVal << "; x: " << maxLoc.x << "; y: " << maxLoc.y << std::endl;

    

    return 0;
}
