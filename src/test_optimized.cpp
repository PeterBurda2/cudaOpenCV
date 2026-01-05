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
#include <fstream>
#include <iomanip> 

int main() {
    omp_set_num_threads(omp_get_num_procs() - 2); 

    
    cv::TickMeter tm;
    // std::cout << "omp_get_num_procs(): " << omp_get_num_procs() << "\n";
    std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << "\n";

    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices available: " << deviceCount << std::endl;

    if (deviceCount <= 0) {
        std::cout << "No CUDA-capable device found or OpenCV built without CUDA." << std::endl;
        return 0;
    }


    std::string imgName = "data/frame-0.tif";
    const int imgNum = 420;
    cv::Mat img;
    cv::Mat imgBlur;
    cv::Size ksize(31, 31);

    double sigma = 11;
    int method = cv::TM_CCOEFF_NORMED;

  
 
    const int h = img.rows;
    const int w = img.cols;

    int cx = w/2;
    int cy = h/2;

    int ox = cx - cx/2;
    int oy = cy - cy/2;


    cv::Rect myROI(ox, oy, cx, cy);
    cv::Mat croppedImg = img(myROI);
    cv::Mat croppedImgBlur;
    cv::GaussianBlur(croppedImg, croppedImgBlur, ksize, sigma);

    const int resWCPU = imgBlur.cols - croppedImgBlur.cols + 1;
    const int resHCPU = imgBlur.rows - croppedImgBlur.rows + 1;

    std::vector<double> maxVal_cpu(imgNum);
    std::vector<cv::Point> maxLoc_cpu(imgNum);

    ////////////////////////////////////////////
                    // CPU PART //
    ////////////////////////////////////////////

    const int nThreads = omp_get_max_threads();
    std::vector<cv::Mat> resPerThread(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        resPerThread[t].create(resHCPU, resWCPU, CV_32F);
    }


    // #pragma omp parallel 
    tm.start();
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < imgNum; i++ ){
        // Assign thread number identification
        const int tid = omp_get_thread_num();
        // std::cout << "Thread number: " << tid << std::endl;
        cv::Mat& cpuRes = resPerThread[tid];

        cv::matchTemplate(imgBlur, croppedImgBlur, cpuRes, method);

        double minV, maxV;
        cv::Point minP, maxP;
        cv::minMaxLoc(cpuRes, &minV, &maxV, &minP, &maxP);

        maxVal_cpu[i] = maxV;
        maxLoc_cpu[i] = maxP;
    }


    std::ofstream csvCPU("results_csv/resultsCPU.csv");
    csvCPU << "iter,x [px],y [px],confidence\n";


    for (int i = 0; i < imgNum; ++i) {
        csvCPU << i << "," << maxLoc_cpu[i].x << "," << maxLoc_cpu[i].y << "," << std::setprecision(4) << std::fixed << maxVal_cpu[i] << "\n";
    }
    csvCPU.close();

    tm.stop();

    std::cout << "CPU matching elapsed time: " << tm.getTimeSec()<< " sec" << std::endl;
    std::cout << "CPU FPS: " << imgNum /tm.getTimeSec() << " FPS" << std::endl;

    tm.reset();


    ////////////////////////////////////////////
                    // GPU PART //
    ////////////////////////////////////////////

    cv::cuda::DeviceInfo dev(0);
    std::cout << "Using device 0: " << dev.name() << std::endl;
    const int resW = w - croppedImgBlur.cols + 1;
    const int resH = h - croppedImgBlur.rows + 1;

        // ---- Pinned host memory (HostMem) ----
    // These give faster H2D/D2H transfers if/when you do them.
    cv::cuda::HostMem h_imgPinned(h, w, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
    cv::cuda::HostMem h_templPinned(croppedImgBlur.rows, croppedImgBlur.cols, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
    cv::cuda::HostMem h_resPinned(resH, resW, CV_32FC1, cv::cuda::HostMem::PAGE_LOCKED);


    imgBlur.copyTo(h_imgPinned.createMatHeader());
    croppedImgBlur.copyTo(h_templPinned.createMatHeader());

    // // // Upload to GPU
    cv::cuda::GpuMat d_img, d_templ, d_res;

    d_img.create(h, w, CV_8UC1);
    d_templ.create(croppedImgBlur.rows, croppedImgBlur.cols, CV_8UC1);
    d_res.create(resH, resW, CV_32FC1);

    cv::cuda::Stream stream;
    
    d_img.upload(h_imgPinned, stream);
    d_templ.upload(h_templPinned, stream);


    cv::Ptr<cv::cuda::TemplateMatching> matcher =
        cv::cuda::createTemplateMatching(CV_8UC1, method); 
    
    // Warm-up 
    matcher->match(d_img, d_templ, d_res, stream);
    stream.waitForCompletion();
    
    double minValGPU = 0.0, maxValGPU = 0.0;
    cv::Point minLocGPU, maxLocGPU;

    std::ofstream csvGPU("results_csv/resultsGPU.csv");
    csvGPU << "iter,x [px],y [px],confidence\n";

    tm.start();
    for(int i = 0; i < imgNum; i++){

        matcher->match(d_img, d_templ, d_res);
        stream.waitForCompletion();
        
        cv::cuda::minMaxLoc(d_res, &minValGPU, &maxValGPU, &minLocGPU, &maxLocGPU);
        csvGPU <<  i << "," << maxLocGPU.x << "," << maxLocGPU.y << "," << std::setprecision(4) << std::fixed << maxValGPU << "\n";
    }

    csvGPU.close();

    tm.stop();

    // d_res.download(h_resPinned, stream);
    // cv::Mat res = h_resPinned.createMatHeader();

    std::cout << "GPU matching elapsed time: " << tm.getTimeSec() << " sec" << std::endl;
    std::cout << "GPU FPS: " << imgNum /tm.getTimeSec() << " FPS" << std::endl;

    // cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // std::cout << "GPU results: confidence: " << maxVal << "; x: " << maxLoc.x << "; y: " << maxLoc.y << std::endl;

    

    return 0;
}
