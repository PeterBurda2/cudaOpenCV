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
    omp_set_num_threads(omp_get_num_procs() - 1); 
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
    const int templtNum = 5;
    double pixel_size = 1.0;
    const int series_number = 1;
    double threshold = 0.75;



    std::vector<cv::Mat> imgSeries(imgNum);
    std::vector<cv::Mat> templateTrack(templtNum);

    // cv::Mat templateTrack = cv::imread("templates/template_1.tiff", cv::IMREAD_UNCHANGED);
    // cv::GaussianBlur(templateTrack, templateTrack, ksize, sigma);

    tm.stop();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < templtNum; ++i) {

        std::string templateName = "templates/template_" + std::to_string(i + 1) + ".tiff";
        cv::Mat imgTemplate = cv::imread(templateName, cv::IMREAD_UNCHANGED);

        cv::GaussianBlur(imgTemplate, imgTemplate, ksize, sigma);
        templateTrack[i] = std::move(imgTemplate);

    }

    tm.stop();

    std::cout << "Template load Time: " << tm.getTimeSec() << std::endl;

    tm.reset();

    tm.start();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < imgNum; ++i) {

        std::string frameName = "series_1/img_1_" + std::to_string(i + 1) + ".tiff";
        cv::Mat img = cv::imread(frameName, cv::IMREAD_UNCHANGED);

        cv::GaussianBlur(img, img, ksize, sigma);
        imgSeries[i] = std::move(img);

    }
    tm.stop();

    std::cout << "Image Series load Time: " << tm.getTimeSec() << std::endl;

    tm.reset();

    // cv::imshow("Image 1", imgSeries[0]);
    // cv::imshow("Image 2", imgSeries[100]);

    // cv::waitKey(0);
    // cv::destroyAllWindows();

    ////////////////////////////////////////////
                    // CPU PART //
    ////////////////////////////////////////////
    const int H = imgSeries[0].rows;
    const int W = imgSeries[0].cols;
    double total_time = 0.0;

    const int nThreads = omp_get_max_threads();

    std::vector<double> maxVal_cpu(imgNum);
    std::vector<cv::Point> maxLoc_cpu(imgNum);
    int method = cv::TM_CCOEFF_NORMED;

    for(int it = 0; it < templtNum; it++){

        maxVal_cpu.clear();
        maxLoc_cpu.clear();
    
        const int h = templateTrack[it].rows;
        const int w = templateTrack[it].cols;

        const int resWCPU = W - w + 1;
        const int resHCPU = H - h + 1;

        std::vector<cv::Mat> resPerThread(nThreads);
        for (int t = 0; t < nThreads; ++t) {
            resPerThread[t].create(resHCPU, resWCPU, CV_32F);
        }


        // #pragma omp parallel 
        tm.start();
        #pragma omp parallel for schedule(static, 1)
        for(int i = 0; i < imgNum; i++ ){
            // Assign thread number identification
            const int tid = omp_get_thread_num();
            // std::cout << "Thread number: " << tid << std::endl;
            cv::Mat& cpuRes = resPerThread[tid];

            cv::matchTemplate(imgSeries[i], templateTrack[it], cpuRes, method);

        
            double minV, maxV;
            cv::Point minP, maxP;
            cv::minMaxLoc(cpuRes, &minV, &maxV, &minP, &maxP);

            maxVal_cpu[i] = maxV;
            maxLoc_cpu[i] = maxP;
        }

        std::string csvCPUName = "results_csv/resultsCPU_" + std::to_string(it + 1) + ".csv";
        std::ofstream csvCPU(csvCPUName);
        // csvCPU << "index,x [px],y [px],confidence\n";
        csvCPU << "index,frame_index,dx [nm],dx [px],dy [nm],dy [px],series,descriptor,descriptor_2,confidence,p_x,p_y\n";

        for (int i = 0; i < imgNum; ++i) {


            double dx = maxLoc_cpu[0].x - maxLoc_cpu[i].x ;
            double dy = maxLoc_cpu[0].y - maxLoc_cpu[i].y ;

            if (maxVal_cpu[i] < threshold){

                std::cout << "Threshold limiter "<< maxVal_cpu[i] << std::endl;
                break;
            }

            csvCPU << i + 1 << "," << std::to_string(series_number)  + "-" + std::to_string(i) << "," 
            << dx * pixel_size << "," << dx << "," 
            << dy * pixel_size << "," << dy << ","
            << std::to_string(series_number) << ","
            <<"S:" + std::to_string(series_number) + " - T:" + std::to_string(it + 1) + " - IMG:" + std::to_string(i + 1) << ","
            <<"S:" + std::to_string(series_number) + " - IMG:" + std::to_string(i + 1) << ","
            <<std::setprecision(5) << std::fixed << maxVal_cpu[i] << "," 
            << maxLoc_cpu[i].x <<","
            << maxLoc_cpu[i].y <<"\n";
            
        }
        csvCPU.close();

        tm.stop();

        std::cout << "CPU matching elapsed time: " << tm.getTimeSec()<< " sec" << std::endl;
        std::cout << "CPU FPS: " << imgNum /tm.getTimeSec() << " FPS" << std::endl;
        total_time += tm.getTimeSec();

        tm.reset();

    }

    std::cout << "Total calculation time: " << total_time << std::endl;

    // ////////////////////////////////////////////
    //                 // GPU PART //
    // ////////////////////////////////////////////

    cv::cuda::DeviceInfo dev(0);
    std::cout << "Using device 0: " << dev.name() << std::endl;

    cv::cuda::Stream stream;
    cv::Ptr<cv::cuda::TemplateMatching> matcher =
        cv::cuda::createTemplateMatching(CV_8UC1, method);
    
    std::vector<cv::cuda::HostMem> h_series(imgNum);
    std::vector<cv::Mat> seriesPinned(imgNum);


    for (int i = 0; i < imgNum; ++i) {
        h_series[i] = cv::cuda::HostMem(H, W, imgSeries[i].type(), cv::cuda::HostMem::PAGE_LOCKED);
        seriesPinned[i] = h_series[i].createMatHeader();
        imgSeries[i].copyTo(seriesPinned[i]);   // one-time copy (NOT inside tracking loop)
    }

    // optional: free pageable copies to save RAM
    imgSeries.clear();
    imgSeries.shrink_to_fit();

    // Upload all templates once
    std::vector<cv::cuda::GpuMat> d_templ(templtNum);
    for (int t = 0; t < templtNum; ++t) {
        // d_templ[t].create(templateTrack[t].rows, templateTrack[t].cols, CV_8UC1);
        d_templ[t].upload(templateTrack[t], stream);
    }

    stream.waitForCompletion();

    // Storage for results
    std::vector<std::vector<double>> maxVal_gpu(templtNum, std::vector<double>(imgNum));
    std::vector<std::vector<cv::Point>> maxLoc_gpu(templtNum, std::vector<cv::Point>(imgNum));

    // Pre allocate result matrices
    std::vector<cv::cuda::GpuMat> d_res(templtNum);
    for (int t = 0; t < templtNum; ++t) {
        int ht = templateTrack[t].rows;
        int wt = templateTrack[t].cols;
        int resH = H - ht + 1;
        int resW = W - wt + 1;
        d_res[t].create(resH, resW, CV_32F);
    }

    // Create device image matrix for upload
    cv::cuda::GpuMat d_img;
    d_img.create(H, W, CV_8UC1);
    tm.reset();


    tm.start();
    for (int i = 0; i < imgNum; ++i) {

        // // pinned -> device (fast / async-capable)
        d_img.upload(seriesPinned[i], stream);

        for (int t = 0; t < templtNum; ++t) {

            matcher->match(d_img, d_templ[t], d_res[t], stream);

            // Make sure match finished before minMaxLoc reads d_res[t]
            stream.waitForCompletion();

            double minV, maxV;
            cv::Point minP, maxP;
            cv::cuda::minMaxLoc(d_res[t], &minV, &maxV, &minP, &maxP);

            maxVal_gpu[t][i] = maxV;
            maxLoc_gpu[t][i] = maxP;
        }
    }

    tm.stop();

    std::cout << "GPU matching elapsed time: " << tm.getTimeSec() << " sec" << std::endl;
    std::cout << "GPU FPS: " << (imgNum * templtNum)/ tm.getTimeSec() << " FPS" << std::endl;



    // const int resW = W - w + 1;
    // const int resH = H - h + 1;


    // cv::cuda::GpuMat d_img, d_templ, d_res;
    // double minValGPU = 0.0, maxValGPU = 0.0;
    // cv::Point minLocGPU, maxLocGPU;

    // d_templ.upload(templateTrack, stream);
    // stream.waitForCompletion();

    // tm.start();
    // for (int i = 0; i < imgNum; ++i) {
    //     int k = i % K;
    //     // std::cout << "k: " << k << std::endl;

    //     // // pageable -> pinned
    //     imgSeries[i].copyTo(ring[k]);

    //     // // pinned -> device (fast / async-capable)
    //     d_img.upload(ring[k], stream);

    //     matcher->match(d_img, d_templ, d_res, stream);
    //     stream.waitForCompletion(); // simplest; for real overlap use events (see below)

    //     cv::cuda::minMaxLoc(d_res, &minValGPU, &maxValGPU, &minLocGPU, &maxLocGPU);
    //     maxVal_gpu[i] = maxValGPU;
    //     maxLoc_gpu[i] = maxLocGPU;

    // }
    // tm.stop();

    // std::cout << "GPU matching elapsed time: " << tm.getTimeSec()<< " sec" << std::endl;
    // std::cout << "GPU FPS: " << imgNum /tm.getTimeSec() << " FPS" << std::endl;



    // std::ofstream csvGPU("results_csv/resultsGPU.csv");
    // csvGPU << "iter,x [px],y [px],confidence\n";
    // for (int i = 0; i < imgNum; ++i) {
    //     csvGPU << i << "," << maxLoc_gpu[i].x << "," << maxLoc_gpu[i].y
    //         << "," << std::setprecision(4) << std::fixed << maxVal_gpu[i] << "\n";
    // }
    // csvGPU.close();

    return 0;
}
