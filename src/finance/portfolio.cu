#include "portfolio.cuh"

#include <math.h>
#include <numeric>
#include <algorithm>

#include <stdexcept>

#include <iostream>
#include <iomanip>

namespace fin
{

__host__ __device__ Portfolio::Portfolio(int size, bool gpu)
{
  this->size = size;
  this->gpu = gpu;
  if (gpu) {
    // portfolio object is going to be used on gpu
    cudaError_t err = cudaMalloc((void **) &(this->assets), sizeof(Asset*) * size);
    check_error(err);
    err = cudaMalloc((void **) &(this->weights), sizeof(float) * size);
    check_error(err);
  } else {
    // portfolio object is going to be used on cpu
    this->assets = new Asset* [size];
    this->weights = new float [size];
  }
}

__host__ __device__ Portfolio::~Portfolio()
{
  if (this->gpu) {
    cudaFree(this->assets);
    cudaFree(this->weights);
  } else {
    delete[] this->assets;
    delete[] this->weights;
  }
}

__host__ void Portfolio::check_error(cudaError_t err)
{
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err)
    << "in " << __FILE__
    << "at line " << __LINE__
    << std::endl;
    exit(EXIT_FAILURE);
  }
}

__host__ __device__ int Portfolio::get_size() const
{
  return this->size;
}

__host__ __device__ Asset** Portfolio::get_assets() const
{
  return this->assets;
}

__host__ __device__ void Portfolio::set_assets(Asset** assets)
{
  for (int i = 0; i < this->size; ++i)
    this->assets[i] = assets[i];
}

__host__ __device__ void Portfolio::set_weights(float* weights)
{
  // check if weights sum to 1
  float sum = 0;
  for (int i = 0; i < this->size; ++i)
    sum += weights[i];
  float eps = 0.01;
  if (sum >= 1 - eps && sum <= 1 + eps)
    for (int i = 0; i < this->size; ++i)
      this->weights[i] = weights[i]; // set weights
  else
    throw std::invalid_argument("Weights must sum to 1");
}


__host__ __device__ float* Portfolio::get_returns(hlp::Date start_date, hlp::Date end_date) const
{
  float* returns[this->size];
  for (int i = 0; i < this->size; ++i)
    returns[i] = this->assets[i]->get_return(start_date, end_date);
  return returns;
}

__host__ __device__ float Portfolio::get_return(hlp::Date start_date, hlp::Date end_date) const
{
  float ret = 0.0;
  for (int i = 0; i < this->size; ++i) {
    float r = this->assets[i]->get_return(start_date, end_date);
    float w = this->weights[i];
    ret += w * r;
  }
  return ret;
}

__host__ __device__ float* Portfolio::get_covariance(hlp::Date start_date, hlp::Date end_date) const
{
  float covariance[this->size * this->size];

  for (int i = 0; i < this->size; ++i) {
    for (int j = 0; j < this->size; ++j) {
      if (j < i) // covariance is symetric, just copy the other half
        covariance[i * this->size + j] = covariance[j * this->size + i];
      else {
        // get dayly returns of assets i and j
        int n;
        float* ri = this->assets[i]->get_returns(start_date, end_date, &n);
        float* rj = this->assets[j]->get_returns(start_date, end_date, &n);

        // get average return of assets i and j
        float ri_avg = 0;
        float rj_avg = 0;
        for (int k = 0; k < n; ++k) {
          ri_avg += ri[k];
          rj_avg += rj[k];
        }
        ri_avg /= n;
        rj_avg /= n;

        // compute covariance between the assets i and j
        covariance[i * this->size + j] = 0;
        for (int k = 0; k < n; ++k)
          covariance[i * this->size + j] += (ri[k] - ri_avg) * (rj[k] - rj_avg);
        covariance[i * this->size + j] /= n;
      }
    }
  }
  return covariance;
}

__host__ __device__ float Portfolio::get_volatility(hlp::Date start_date, hlp::Date end_date) const
{
  float vol = 0;
  float* cov = this->get_covariance(start_date, end_date);
  for (int i = 0; i < this->size; ++i) {
    for (int j = 0; j < this->size; ++j)
    {
      float wi = this->weights[i];
      float wj = this->weights[j];
      vol += wi * wj * cov[i * this->size + j];
    }
  }
  return sqrtf(vol);
}

__host__ __device__ float Portfolio::get_sharp(hlp::Date start_date, hlp::Date end_date) const
{
  float ret = this->get_return(start_date, end_date);
  float vol = this->get_volatility(start_date, end_date);
  return ret / vol;
}

__host__ void Portfolio::print_weights() const
{
  int size = this->assets.size();
  for (int i = 0; i < size; ++i) {
    float w = this->weights[i];
    std::cout << std::fixed << std::setprecision(2) << w << "|";
  }
  std::cout << std::endl;
}

}
