#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cstdio>
#include <cstdlib>

#include "asset.cuh"
#include "../helpers/date.cuh"

namespace fin
{
class Portfolio
{
public:
  CUDA_CALLABLE_MEMBER Portfolio(int size);
  CUDA_CALLABLE_MEMBER ~Portfolio();

  // setters
  CUDA_CALLABLE_MEMBER  void set_assets(Asset** assets);
  CUDA_CALLABLE_MEMBER  void set_weights(float* weights);

  // getters
  CUDA_CALLABLE_MEMBER  Asset** get_assets() const;
  CUDA_CALLABLE_MEMBER  float* get_weights() const;
  CUDA_CALLABLE_MEMBER  int get_size() const;

  // helpers to find optimal portfolio
  CUDA_CALLABLE_MEMBER  float* get_returns(hlp::Date start_date, hlp::Date end_date) const;
  CUDA_CALLABLE_MEMBER  float* get_covariance(hlp::Date start_date, hlp::Date end_date) const;

  // metrics getters
  CUDA_CALLABLE_MEMBER  float get_return(hlp::Date start_date, hlp::Date end_date) const;
  CUDA_CALLABLE_MEMBER  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;
  CUDA_CALLABLE_MEMBER  float get_sharp(hlp::Date start_date, hlp::Date end_date) const;

  void print_weights() const;

private:
  int size;
  Asset** assets;
  float* weights;

};

}
