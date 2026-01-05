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
    cv::setNumThreads(1);   // before the parallel region

    
    cv::TickMeter tm;
    // std::cout << "omp_get_num_procs(): " << omp_get_num_procs() << "\n";
    std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << "\n";

    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices available: " << deviceCount << std::endl;

    if (deviceCount <= 0) {
        std::cout << "No CUDA-capable device found or OpenCV built without CUDA." << std::endl;
        return 0;
    }


    // std::string imgName = "data/frame-0.tif";


    // // Gaussian blur parameters
    double sigma = 11;
    cv::Size ksize(31, 31);
    const int imgNum = 420;


    std::vector<cv::Mat> imgSeries(imgNum);
    cv::Mat templateTrack = cv::imread("templates/t_1.tif", cv::IMREAD_UNCHANGED);
    cv::GaussianBlur(templateTrack, templateTrack, ksize, sigma);


    tm.start();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < imgNum; ++i) {

        std::string frameName = "series_1/frame_" + std::to_string(i) + ".tif";
        cv::Mat img = cv::imread(frameName, cv::IMREAD_UNCHANGED);

        cv::GaussianBlur(img, img, ksize, sigma);
        imgSeries[i] = std::move(img);

    }
    tm.stop();

    std::cout << "Load Time: " << tm.getTimeSec() << std::endl;
    const int H = imgSeries[0].rows;
    const int W = imgSeries[0].cols;
    const int h = templateTrack.rows;
    const int w = templateTrack.cols;

    // cv::imshow("Image 1", imgSeries[0]);
    // cv::imshow("Image 2", imgSeries[100]);

    // cv::waitKey(0);
    // cv::destroyAllWindows();

    ////////////////////////////////////////////
                    // CPU PART //
    ////////////////////////////////////////////
    const int resWCPU = W - w + 1;
    const int resHCPU = H - h + 1;

    const int nThreads = omp_get_max_threads();
    std::vector<cv::Mat> resPerThread(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        resPerThread[t].create(resHCPU, resWCPU, CV_32F);
    }


    std::vector<double> maxVal_cpu(imgNum);
    std::vector<cv::Point> maxLoc_cpu(imgNum);
    int method = cv::TM_CCOEFF_NORMED;


    // #pragma omp parallel 
    tm.start();
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < imgNum; i++ ){
        // Assign thread number identification
        const int tid = omp_get_thread_num();
        // std::cout << "Thread number: " << tid << std::endl;
        cv::Mat& cpuRes = resPerThread[tid];

        cv::matchTemplate(imgSeries[i], templateTrack, cpuRes, method);

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


    // ////////////////////////////////////////////
    //                 // GPU PART //
    // ////////////////////////////////////////////

    cv::cuda::DeviceInfo dev(0);
    std::cout << "Using device 0: " << dev.name() << std::endl;
    cv::cuda::Stream stream;
    cv::Ptr<cv::cuda::TemplateMatching> matcher =
        cv::cuda::createTemplateMatching(CV_8UC1, method);
    
    double minValGPU = 0.0, maxValGPU = 0.0;
    cv::Point minLocGPU, maxLocGPU;

    const int resW = W - w + 1;
    const int resH = H - h + 1;


    const int K = 32;  // ring size (8..32 typical)
    std::vector<cv::cuda::HostMem> h_ring(K);
    std::vector<cv::Mat> ring(K);


    for (int k = 0; k < K; ++k) {
        h_ring[k] = cv::cuda::HostMem(H, W, imgSeries[0].type(), cv::cuda::HostMem::PAGE_LOCKED);
        ring[k] = h_ring[k].createMatHeader();
    }

    cv::cuda::GpuMat d_img, d_templ, d_res;


    for (int i = 0; i < imgNum; ++i) {
        int k = i % K;
        std::cout << "k: " << k << std::endl;

        // // pageable -> pinned
        imgSeries[i].copyTo(ring[k]);

        // // pinned -> device (fast / async-capable)
        d_img.upload(ring[k], stream);

        matcher->match(d_img, d_templ, d_res, stream);
        stream.waitForCompletion(); // simplest; for real overlap use events (see below)

        // cv::cuda::minMaxLoc(d_res, &minValGPU, &maxValGPU, &minLocGPU, &maxLocGPU);

    }


    // ---- Pinned host memory (HostMem) ----
    // cv::cuda::HostMem h_imgPinned(H, W, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
    // cv::cuda::HostMem h_templPinned(h, w, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
    // cv::cuda::HostMem h_resPinned(resH, resW, CV_32FC1, cv::cuda::HostMem::PAGE_LOCKED);


    // imgBlur.copyTo(h_imgPinned.createMatHeader());
    // croppedImgBlur.copyTo(h_templPinned.createMatHeader());

    // // // // Upload to GPU
    // cv::cuda::GpuMat d_img, d_templ, d_res;

    // d_img.create(h, w, CV_8UC1);
    // d_templ.create(croppedImgBlur.rows, croppedImgBlur.cols, CV_8UC1);
    // d_res.create(resH, resW, CV_32FC1);

    
    // d_img.upload(h_imgPinned, stream);
    // d_templ.upload(h_templPinned, stream);


    // cv::Ptr<cv::cuda::TemplateMatching> matcher =
    //     cv::cuda::createTemplateMatching(CV_8UC1, method); 
    
    // // Warm-up 
    // matcher->match(d_img, d_templ, d_res, stream);
    // stream.waitForCompletion();
    
    // double minValGPU = 0.0, maxValGPU = 0.0;
    // cv::Point minLocGPU, maxLocGPU;

    // std::ofstream csvGPU("results_csv/resultsGPU.csv");
    // csvGPU << "iter,x [px],y [px],confidence\n";

    // tm.start();
    // for(int i = 0; i < imgNum; i++){

    //     matcher->match(d_img, d_templ, d_res);
    //     stream.waitForCompletion();
        
    //     cv::cuda::minMaxLoc(d_res, &minValGPU, &maxValGPU, &minLocGPU, &maxLocGPU);
    //     csvGPU <<  i << "," << maxLocGPU.x << "," << maxLocGPU.y << "," << std::setprecision(4) << std::fixed << maxValGPU << "\n";
    // }

    // csvGPU.close();

    // tm.stop();

    // // d_res.download(h_resPinned, stream);
    // // cv::Mat res = h_resPinned.createMatHeader();

    // std::cout << "GPU matching elapsed time: " << tm.getTimeSec() << " sec" << std::endl;
    // std::cout << "GPU FPS: " << imgNum /tm.getTimeSec() << " FPS" << std::endl;

    // cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // std::cout << "GPU results: confidence: " << maxVal << "; x: " << maxLoc.x << "; y: " << maxLoc.y << std::endl;

    

    return 0;
}
