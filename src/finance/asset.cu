#include "asset.cuh"

#include <math.h>
#include <numeric>
#include <algorithm>

#include <stdio.h>


namespace fin
{

CUDA_CALLABLE_MEMBER
Asset::Asset(int id)
{
  this->id = id;
  this->size = 0;
  this->closes = 0;
}

CUDA_CALLABLE_MEMBER
Asset::Asset()
{
  this->id = -1;
  this->size = 0;
  this->closes = 0;
}

CUDA_CALLABLE_MEMBER
Asset::Asset(const Asset& asset)
{
    this->id = asset.id;
    this->size = asset.size;
    this->closes = 0;
    if (asset.closes != 0)
    {
        // deep copy asset closes
        this->closes = new Close [asset.size];
        for (int i = 0; i < asset.size; ++i)
            this->closes[i] = asset.closes[i];
    }
}

CUDA_CALLABLE_MEMBER
Asset Asset::operator=(const Asset& asset)
{
    this->id = asset.id;
    this->size = asset.size;
    this->closes = 0;
    if (asset.closes != 0)
    {
        // deep copy asset closes
        this->closes = new Close [asset.size];
        for (int i = 0; i < asset.size; ++i)
            this->closes[i] = asset.closes[i];
    } else 
        printf("Hahaaa = \n");
    return *this;
}

    
CUDA_CALLABLE_MEMBER
Asset::~Asset()
{
    if (this->closes != 0) 
    {
        delete [] this->closes;
        this->closes = 0;
    }
}


void Asset::set_closes(std::vector<Close> closes)
{
  if (closes.size() == 0) 
  {
      printf("closes is empty\n");
      return;
  }
  // sort closes by date
  this->sort_closes(closes);
  // set closes size
  this->size = closes.size();
  // delete old closes if existing
  if (this->closes != 0)
  {
      delete [] this->closes;
      this->closes = 0;
  }
  // allocate closes
  this->closes = new Close[this->size];
  // set closes
  for (int i = 0; i < this->size; ++i)
    this->closes[i] = closes[i];
}

CUDA_CALLABLE_MEMBER
void Asset::set_id(int id)
{
  if (id > 0)
    this->id = id;
}

CUDA_CALLABLE_MEMBER
int Asset::get_id()
{
  return this->id;
}

CUDA_CALLABLE_MEMBER
Close* Asset::get_closes(int *n) const
{
  *n = this->size;
  // allocate memory for closes
  Close* closes = new Close[*n];
  // set closes
  for (int i = 0; i < *n; ++i)
    closes[i] = this->closes[i];
  return closes;
}

CUDA_CALLABLE_MEMBER
Close* Asset::get_closes(hlp::Date start_date, hlp::Date end_date,
        int *n) const
{
  int start = 0;
  int end = -1;
  // get period (start -> end)
  for(int i = 0; i < this->size; ++i)
  {
      if (this->closes[i].date == start_date)
          start = i;
      if (this->closes[i].date == end_date) 
      {
          end = i;
          continue;
      }
  }
  // set size of return array
  *n = end - start;
  if (*n <= 0)
  {
      printf("Woow, couldn't find closes with theses dates\n");
      return 0;
  }
  // allocate memory for closes
  Close* closes = new Close[*n];
  // set closes
  for (int i = 0; i < *n; ++i)
      closes[i] = this->closes[i + start];
  return closes;
}

CUDA_CALLABLE_MEMBER
float Asset::get_return() const
{
  float v1 = this->closes[0].value;
  float v2 = this->closes[this->size - 1].value;
  return (v2 - v1) / v1;
}

CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
float* Asset::get_returns(int *n) const
{
  *n = this->size - 1;
  // allocate memory for returns
  float* returns = new float[*n];
  // compute all daily returns on that period
  for (int i = 0; i < *n; ++i) {
      float v1 = this->closes[i].value;
      float v2 = this->closes[i + 1].value;
      returns[i] = (v2 - v1) / v1;
  }

  return returns;
}

CUDA_CALLABLE_MEMBER
float* Asset::get_returns(hlp::Date start_date, hlp::Date end_date,
        int* n) const
{
  // get asset closes on this period (start->end)
  Close* closes = this->get_closes(start_date, end_date, n);
  // set n to returns size (closes - 1: it's dayly return)
  *n -= 1;
  // allocate memory for returns
  float* returns = new float[*n];
  // compute all daily returns on that period
  for (int i = 0; i < *n; ++i) {
    float v1 = closes[i].value;
    float v2 = closes[i + 1].value;
    returns[i] = (v2 - v1) / v1;
  }
  // free memory reserved for closes
  delete[] closes;
  return returns;
}

CUDA_CALLABLE_MEMBER
float Asset::get_volatility() const
{
  int n;
  // get daily returns
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
  // free memory of daily returns
  delete[] returns;
  // return volatility: sqrt(variance)
  return sqrtf(var);
}

CUDA_CALLABLE_MEMBER
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
  // free memory of daily returns
  delete[] returns;
  // return volatility: sqrt(variance)
  return sqrtf(var);
}

CUDA_CALLABLE_MEMBER 
float Asset::get_sharp() const
{
    float ret = this->get_return();
    float vol = this->get_volatility();
    return ret / vol;
}
  
CUDA_CALLABLE_MEMBER 
float Asset::get_sharp(hlp::Date start_date, hlp::Date end_date) const
{
    float ret = this->get_return(start_date, end_date);
    float vol = this->get_volatility(start_date, end_date);
    return ret / vol;
}

// sort using a custom function object
bool date_less(Close a, Close b)
{
  return a.date < b.date;
}

// sort closes by date
void Asset::sort_closes(std::vector<Close> closes)
{
  std::sort(closes.begin(), closes.end(), date_less);
}

}
