#include "date.hpp"

namespace hlp
{

Date::Date(std::string str_date)
{
  this->year = std::stoi(str_date.substr(0, 4));
  this->month = std::stoi(str_date.substr(5, 2));
  this->day = std::stoi(str_date.substr(8, 2));
}

int Date::get_day() const
{
  return this->day;
}

int Date::get_month() const
{
  return this->month;
}

int Date::get_year() const
{
  return this->year;
}

void Date::set_day(int day)
{
  if (day > 0 && day < 32)
    this->day = day;
}

void Date::set_month(int month)
{
  if (month > 0 and month < 13)
    this->month = month;
}
void Date::set_year(int year) {
  if (year > 1900 && year < 2100)
    this->year = year;
}

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


bool Date::operator>(const Date& d) const
{
  return !(*this < d);
}

bool Date::operator==(const Date& d) const
{
  return this->year == d.get_year() && this->month == d.get_month() && this->day == d.get_day();
}

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
