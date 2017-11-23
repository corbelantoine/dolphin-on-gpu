#include "asset.hpp"

#include <math.h>
#include <numeric>
#include <algorithm>

#include <cstdio>
#include <cstdlib>

namespace fin
{

Asset::Asset(int id)
{
  this->id = id;
  this->closes = 0;
}

Asset::~Asset()
{
  if (this->closes != 0)
    delete[] this->closes;
}

void Asset::set_closes(std::vector<Close> closes)
{
  // sort closes by date
  this->sort_closes(closes);
  // set closes size
  this->size = this->closes.size();
  // allocate closes
  this->closes = new Close[this->size];
  // set closes
  for (int i = 0; i < this->size; ++i)
    this->closes[i] = closes[i];
}

Close* Asset::get_closes() const
{
  return this->closes;
}

Close* Asset::get_closes(hlp::Date start_date, hlp::Date end_date, int *n) const
{
  int start, end;
  for(int i = 0; i < this->size; ++i) {
    if (this->closes[i].date == start_date)
      start = i;
    if (this->closes[i].date == end_date) {
      end = i;
      continue;
    }
  }
  *n = end - start;
  Close closes[*n];
  for (int i = 0; i < *n; ++i)
    closes[i] = this->closes[i + start];
  return closes;
}

float Asset::get_return() const
{
  float v1 = this->closes[0].value;
  float v2 = this->closes[this->size - 1].value;
  return (v2 - v1) / v1;
}

float Asset::get_return(hlp::Date start_date, hlp::Date end_date) const
{
  float v1, v2;
  for (int i = 0; i < this->size; ++i) {
    Close close = this->closes[i];
    if (close.date == start_date)
      v1 = close.value;
    if (close.date == end_date)
      v2 = close.value;
  }
  return (v2 - v1) / v1;
}

float* Asset::get_returns(int *n) const
{
  *n = this->size - 1;
  // compute all dayly returns on that period
  float returns[*n];
  for (int i = 0; i < *n; ++i) {
    float v1 = this->closes[i].value;
    float v2 = this->closes[i + 1].value;
    returns[i] = (v2 - v1) / v1;
  }

  return returns;
}

__host__ __device__ float* Asset::get_returns(hlp::Date start_date, hlp::Date end_date, int* n) const
{
  // get asset closes on this period (start->end)
  Close* closes = this->get_closes(start_date, end_date, n);
  // set n to returns size (closes - 1: it's dayly return)
  *n -= 1;
  // compute all dayly returns on that period
  float returns[*n];
  for (int i = 0; i < *n; ++i) {
    float v1 = closes[i].value;
    float v2 = closes[i + 1].value;
    returns[i] = (v2 - v1) / v1;
  }

  return returns;
}

float Asset::get_volatility() const
{
  int n;
  // get dayly returns
  float* returns = this->get_returns(&n);
  // compute average return
  float avg = 0;
  for (int i = 0; i < n; ++i)
    avg += returns[i];
  avg /= n;
  // compute variance
  float var = 0;
  for (int i = 0; i < n; ++i)
    var += pow(returns[i] - avg, 2);
  var /= n;
  // return volatility: sqrt(variance)
  return sqrtf(var);
}

float Asset::get_volatility(hlp::Date start_date, hlp::Date end_date) const
{
  int n;
  // get dayly returns
  float* returns = this->get_returns(start_date, end_date, &n);
  // compute average return
  float avg = 0;
  for (int i = 0; i < n; ++i)
    avg += returns[i];
  avg /= n;
  // compute variance
  float var = 0;
  for (int i = 0; i < n; ++i)
    var += pow(returns[i] - avg, 2);
  var /= n;
  // return volatility: sqrt(variance)
  return sqrtf(var);
}

// sort using a custom function object
bool date_less(Close a, Close b)
{
  return a.date < b.date;
}

void Asset::sort_closes(std::vector<Close> closes)
{
  std::sort(closes.begin(), closes.end(), date_less);
}

}
