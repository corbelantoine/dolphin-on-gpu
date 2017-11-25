#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cstdio>
#include <cstdlib>

#include <vector>

#include "../helpers/date.cuh"

namespace fin
{

struct Close
{
  hlp::Date date;
  float value;
};

class Asset
{
public:
  CUDA_CALLABLE_MEMBER Asset(int id);
  CUDA_CALLABLE_MEMBER Asset();
  CUDA_CALLABLE_MEMBER ~Asset();

  void set_closes(std::vector<Close> closes);
  CUDA_CALLABLE_MEMBER void set_id(int id);

  CUDA_CALLABLE_MEMBER Close* get_closes(int *n) const;
  CUDA_CALLABLE_MEMBER Close* get_closes(hlp::Date start_date, hlp::Date end_date, int *n) const;

  // methods for asset evaluation
  // get the return of an asset
  CUDA_CALLABLE_MEMBER float get_return() const;
  CUDA_CALLABLE_MEMBER float get_return(hlp::Date start_date, hlp::Date end_date) const;
  // get the volatility of an asset
  CUDA_CALLABLE_MEMBER float get_volatility() const;
  CUDA_CALLABLE_MEMBER float get_volatility(hlp::Date start_date, hlp::Date end_date) const;

  // methods for optimizing portfolio
  CUDA_CALLABLE_MEMBER float* get_returns(int* n) const;
  CUDA_CALLABLE_MEMBER float* get_returns(hlp::Date start_date, hlp::Date end_date, int* n) const;

private:
  int id;
  int size;
  Close* closes;

  void sort_closes(std::vector<Close> closes);
};

}
