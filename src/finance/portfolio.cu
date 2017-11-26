#include "portfolio.cuh"

#include <math.h>
#include <numeric>
#include <algorithm>

#include <stdexcept>

#include <iostream>
#include <iomanip>

#include <stdio.h>

namespace fin
{

CUDA_CALLABLE_MEMBER
Portfolio::Portfolio()
{
  int size = 20;
  this->size = size;
  this->assets = new Asset* [size];
  this->weights = new float [size];
}


CUDA_CALLABLE_MEMBER
Portfolio::Portfolio(int size)
{
  this->size = size;
  this->assets = new Asset* [size];
  this->weights = new float [size];
}

CUDA_CALLABLE_MEMBER
Portfolio::Portfolio(const Portfolio& portfolio)
{
    this->size = portfolio.size;
    this->assets = new Asset* [this->size];
    this->weights = new float [this->size];
    // deep copy assets and weights
    for (int i = 0; i < this->size; ++i)
    {
        this->assets[i] = portfolio.assets[i];
        this->weights[i] = portfolio.weights[i];
    }
}
    
CUDA_CALLABLE_MEMBER
Portfolio::~Portfolio()
{
  delete[] this->assets;
  delete[] this->weights;
}

CUDA_CALLABLE_MEMBER
Portfolio Portfolio::operator=(const Portfolio& portfolio)
{
    this->size = portfolio.size;
    this->assets = new Asset* [this->size];
    this->weights = new float [this->size];
    // deep copy assets and weights
    for (int i = 0; i < this->size; ++i)
    {
        this->assets[i] = portfolio.assets[i];
        this->weights[i] = portfolio.weights[i];
    }
    return *this;
}

CUDA_CALLABLE_MEMBER
int Portfolio::get_size() const
{
  return this->size;
}

CUDA_CALLABLE_MEMBER
Asset** Portfolio::get_assets() const
{
  return this->assets;
}

CUDA_CALLABLE_MEMBER
void Portfolio::set_assets(Asset** assets)
{
  for (int i = 0; i < this->size; ++i)
    this->assets[i] = assets[i];
}

CUDA_CALLABLE_MEMBER
void Portfolio::set_weights(float* weights)
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
    printf("Weights must sum to 1\n");
}


CUDA_CALLABLE_MEMBER
float* Portfolio::get_returns(hlp::Date start_date, hlp::Date end_date) const
{
  // returns a list of return for each asset
  // allocate memory for daily returns
  float* returns = new float[this->size];
  // set daily returns
  for (int i = 0; i < this->size; ++i)
    returns[i] = this->assets[i]->get_return(start_date, end_date);
  return returns;
}

CUDA_CALLABLE_MEMBER
float Portfolio::get_return(hlp::Date start_date, hlp::Date end_date) const
{
  float ret = 0.0;
  for (int i = 0; i < this->size; ++i) {
    float r = this->assets[i]->get_return(start_date, end_date);
    float w = this->weights[i];
    ret += w * r;
  }
  return ret;
}

CUDA_CALLABLE_MEMBER
float* Portfolio::get_covariance(hlp::Date start_date, hlp::Date end_date) const
{
  float* covariance = new float[this->size * this->size];
  for (int i = 0; i < this->size; ++i) {
    for (int j = 0; j < this->size; ++j) {
      if (j < i) // covariance is symetric, just copy the other half
        covariance[i * this->size + j] = covariance[j * this->size + i];
      else {
        // get daily returns of assets i and j
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

        // freeing ri and rj
        delete[] ri;
        delete[] rj;
      }
    }
  }
  return covariance;
}

CUDA_CALLABLE_MEMBER
float Portfolio::get_volatility(hlp::Date start_date, hlp::Date end_date) const
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
  //free covariance
  delete[] cov;
  return sqrtf(vol);
}

CUDA_CALLABLE_MEMBER
float Portfolio::get_sharp(hlp::Date start_date, hlp::Date end_date) const
{
  float ret = this->get_return(start_date, end_date);
  float vol = this->get_volatility(start_date, end_date);
  return ret / vol;
}

void Portfolio::print_weights() const
{
  for (int i = 0; i < this->size; ++i) {
    float w = this->weights[i];
    std::cout << std::fixed << std::setprecision(2) << w << "|";
  }
  std::cout << std::endl;
}

}
