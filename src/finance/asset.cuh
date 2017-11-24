#pragma once

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
  Asset(int id);
  ~Asset();

  void set_closes(std::vector<Close> closes);

  Close* get_closes(int *n) const;
__host__ __device__  Close* get_closes(hlp::Date start_date, hlp::Date end_date, 
          int *n) const;

  float get_return() const;
  float get_return(hlp::Date start_date, hlp::Date end_date) const;
__host__ __device__ float* get_returns(int* n) const;
__host__ __device__ float* get_returns(hlp::Date start_date, hlp::Date end_date,
          int* n) const;
__host__ __device__ float get_volatility() const;
__host__ __device__ float get_volatility(hlp::Date start_date, hlp::Date end_date) const;

private:
  int id;
  int size;
  Close* closes;

  void sort_closes(std::vector<Close> closes);
};

}
