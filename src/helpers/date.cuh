#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <stdio.h>
#include <iostream>


namespace hlp
{

class Date
{

public:
  Date(char* str_date);
  Date() = default;
  ~Date() = default;

  CUDA_CALLABLE_MEMBER int get_day() const;
  CUDA_CALLABLE_MEMBER int get_month() const;
  CUDA_CALLABLE_MEMBER int get_year() const;

  CUDA_CALLABLE_MEMBER void set_day(int day);
  CUDA_CALLABLE_MEMBER void set_month(int month);
  CUDA_CALLABLE_MEMBER void set_year(int year);

  CUDA_CALLABLE_MEMBER bool operator<(const Date& d) const;
  CUDA_CALLABLE_MEMBER bool operator>(const Date& d) const;
  CUDA_CALLABLE_MEMBER bool operator<=(const Date& d) const;
  CUDA_CALLABLE_MEMBER bool operator>=(const Date& d) const;
  CUDA_CALLABLE_MEMBER bool operator==(const Date& d) const;
  CUDA_CALLABLE_MEMBER bool operator!=(const Date& d) const;

private:
  int year;
  int month;
  int day;
};

std::ostream& operator<<(std::ostream& os, const Date& d);

}
