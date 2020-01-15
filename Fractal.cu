#include "Fractal.cuh"
#include "device_launch_parameters.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

__device__ bool julian(int x, int y, int nRows, int nCols, float rc, float imc, float scaleFactor)
{

	float jx = scaleFactor * ((float)(nCols / 2) - x) / (nCols / 2);
	float jy = scaleFactor * ((float)(nRows / 2) - y) / (nRows / 2);

	cudaComplex c(rc, imc);
	cudaComplex z(jx, jy);

	int iteration = 150;
	bool isFrct = true;
	for (int i = 0; i < iteration; ++i)
	{
		z = z * z + c;
		if (z.magnitude2() > 1000)
		{
			isFrct = false;
			break;
		}
	}
	return isFrct;
}

__global__ void FractalDraw(uchar* data, int nRows, int nCols, float rc, float imc, float scaleFactor)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	int offset = x + y * blockDim.x*gridDim.x;
	//

	if (x < nCols && y < nRows)
	{

		bool isFrct = julian(x, y, nRows, nCols, rc, imc, scaleFactor);


		data[offset * 3 + 0] = 120;
		data[offset * 3 + 1] = 65;
		data[offset * 3 + 2] = 255 * isFrct;
	}

}