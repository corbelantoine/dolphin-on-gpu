#pragma once

#include <iostream>

#include <cstdio>
#include <cstdlib>

namespace hlp
{

class Date
{

public:
  Date(std::string str_date);
  Date() = default;
  ~Date() = default;

  __host__ __device__ int get_day() const;
  __host__ __device__ int get_month() const;
  __host__ __device__ int get_year() const;

  __host__ __device__ void set_day(int day);
  __host__ __device__ void set_month(int month);
  __host__ __device__ void set_year(int year);

  __host__ __device__ bool operator<(const Date& d) const;
  __host__ __device__ bool operator>(const Date& d) const;
  __host__ __device__ bool operator<=(const Date& d) const;
  __host__ __device__ bool operator>=(const Date& d) const;
  __host__ __device__ bool operator==(const Date& d) const;
  __host__ __device__ bool operator!=(const Date& d) const;

private:
  int year;
  int month;
  int day;
};

std::ostream& operator<<(std::ostream& os, const Date& d);

}
