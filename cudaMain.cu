#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include "Fractal.cuh"


void static handler_error(cudaError_t error, const char* file, int line)
{
	if (error != cudaSuccess)
	{
		const char* errorString = cudaGetErrorString(error);
		std::cout << errorString << " " << " line: " << line << " file: " << file << std::endl;
		system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (handler_error( err, __FILE__, __LINE__ ))






int main()
{
	std::string info = "z=z*z+c\nincrease real part, press q or Q\ndecrease real part, press a or A\n"
		"increase imaginary part, press w or W\ndecrease imaginary part, press s or S\n";
	std::cout << info << std::endl;
	//cfg
	size_t nRows = 640;
	size_t nCols = 640;

	dim3 threads(20, 20);
	dim3 blocks(32, 32);
	//

	cv::Mat plane = cv::Mat::zeros(nRows, nCols, CV_8UC3);
	uchar* dev_data = nullptr;

	HANDLE_ERROR(cudaMalloc((void**)&dev_data, sizeof(uchar)*nRows*nCols * 3));

	float r = 0;
	float im = 0;

	float dR = 0.001;
	float dIm = 0.001;
	
	float scaleFactor = 2;
	

	std::cout << "start value R="; std::cin >> r;
	std::cout << "start value Im="; std::cin >> im;

	std::cout << "dR="; std::cin >> dR;
	std::cout << "dIm="; std::cin >> dIm;

	std::cout << "set scale: "; std::cin >> scaleFactor;

	while (true)
	{

		/*r += dR;
		im += dIm;*/
		FractalDraw << <blocks, threads >> > (dev_data, nRows, nCols, r, im, scaleFactor);
		HANDLE_ERROR(cudaMemcpy(plane.data, dev_data, sizeof(uchar)*nRows*nCols * 3, cudaMemcpyDeviceToHost));


		cv::imshow("img", plane);
		int codeKey = cv::waitKey();
		
		switch (codeKey)
		{
		case 113:
		case 81:// increase real part, press q or Q
			r += dR;
			break;
		case 97:
		case 65://decrease real part, press a or A
			r -= dR;
			break;
		case 119:
		case 87:// increase imaginary part, press w or W
			im += dIm;
			break;
		case 115:
		case 83://decrease imaginary part, press s or S
			im -= dIm;
			break;
		default:
			break;
		}
		std::cout <<"r="<< r << " " <<"im="<< im << std::endl;

		if (codeKey == 27)
			break;
	}

	cudaFree(dev_data);
}