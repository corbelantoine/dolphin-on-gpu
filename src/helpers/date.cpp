#include "date.hpp"

namespace hlp
{

Date::Date(std::string str_date)
{
  this.year = std::stoi(str_date.substr(0, 4));
  this.month = std::stoi(str_date.substr(5, 2));
  this.day = std::stoi(str_date.substr(8, 2));
}

int Date::get_day()
{
  return this.day;
}

int Date::get_month()
{
  return this.month;
}

int Date::get_year()
{
  return this.year;
}

void Date::set_day(int day)
{
  if (day > 0 && day < 32)
    this.day = day;
}

void Date::set_month(int month)
{
  if (month > 0 and month < 13)
    this.month = month;
}
void Date::set_year(int year) {
  if (year > 1900 && year < 2100)
    this.year = year;
}

bool Date::operator<(const Date& d)
{
  // check year
  if (this.year < d.year)
    return true;
  else if (this.year > d.year)
    return false;
  // check month
  if (this.month < d.month)
    return true;
  else if (this.month > d.month)
    return false;
  // check day
  if (this.day < d.day)
    return true;
  return false;
}


bool Date::operator>(const Date& d)
{
  return ! this < d;
}

bool Date::operator==(const Date& d)
{
  return this.year == d.year && this.month == d.month && this.day = d.day;
}

bool Date::operator!=(const Date& d)
{
  return ! this == d;
}

std::ostream& operator<<(std::ostream& os, const Date& d)
{
  // write obj to stream
  std::cout << d.year << "-"
            << d.month << "-"
            << d.day
  return os;
}

}
