#include "date.cuh"

namespace hlp
{

CUDA_CALLABLE_MEMBER
Date::Date(char* str_date)
{
  if (3 != sscanf(string, "%d-%d-%d;", &(this->year), &(this->month), &(this->day))) {
    printf("error scanning date. format: YYYY-MM-DD\n");
  }
}

CUDA_CALLABLE_MEMBER
int Date::get_day() const
{
  return this->day;
}

CUDA_CALLABLE_MEMBER
int Date::get_month() const
{
  return this->month;
}

CUDA_CALLABLE_MEMBER
int Date::get_year() const
{
  return this->year;
}

CUDA_CALLABLE_MEMBER
void Date::set_day(int day)
{
  if (day > 0 && day < 32)
    this->day = day;
}

CUDA_CALLABLE_MEMBER
void Date::set_month(int month)
{
  if (month > 0 and month < 13)
    this->month = month;
}
CUDA_CALLABLE_MEMBER
void Date::set_year(int year) {
  if (year > 1900 && year < 2100)
    this->year = year;
}

CUDA_CALLABLE_MEMBER
bool Date::operator<(const Date& d) const
{
  // check year
  if (this->year < d.get_year())
    return true;
  else if (this->year > d.get_year())
    return false;
  // check month
  if (this->month < d.get_month())
    return true;
  else if (this->month > d.get_month())
    return false;
  // check day
  if (this->day < d.get_day())
    return true;
  return false;
}


CUDA_CALLABLE_MEMBER
bool Date::operator>(const Date& d) const
{
  return !(*this < d);
}

CUDA_CALLABLE_MEMBER
bool Date::operator==(const Date& d) const
{
  return this->year == d.get_year() && this->month == d.get_month() && this->day == d.get_day();
}

CUDA_CALLABLE_MEMBER
bool Date::operator<=(const Date& d) const
{
  return *this < d || *this == d;
}

CUDA_CALLABLE_MEMBER
bool Date::operator>=(const Date& d) const
{
  return *this > d || *this == d;
}

CUDA_CALLABLE_MEMBER
bool Date::operator!=(const Date& d) const
{
  return !(*this == d);
}

std::ostream& operator<<(std::ostream& os, const Date& d)
{
  // write obj to stream
  std::cout << d.get_year() << "-"
            << d.get_month() << "-"
            << d.get_day();
  return os;
}

}
