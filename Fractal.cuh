#pragma once
#ifndef _FRACTAL_H_
#define _FRACTAL_H_

#include "cuda_runtime.h"
#include <stdio.h>

typedef unsigned char uchar;
struct cudaComplex
{
	float r;
	float im;

	__device__ cudaComplex() :r(0), im(0) {}
	__device__ cudaComplex(float _r, float _im) : r(_r), im(_im) {}
	__device__ float magnitude2()noexcept
	{
		return r * r + im * im;
	}
	__device__ cudaComplex operator*(const cudaComplex& _other)
	{
		return cudaComplex(r*_other.r - im * _other.im, im*_other.r + r * _other.im);
	}
	__device__ cudaComplex operator+(const cudaComplex& _other)
	{
		return cudaComplex(r + _other.r, im + _other.im);
	}
};

__device__ bool julian(int x, int y, int nRows, int nCols, float rc, float imc, float scaleFactor);

__global__ void FractalDraw(uchar* data, int nRows, int nCols, float rc, float imc, float scaleFactor);
#endif // !_FRACTAL_H_
