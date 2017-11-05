#pragma once

#include <iostream>

namespace hlp
{

class Date
{

public:
  Date(std::string str_date);
  ~Date() = default;

  int get_day();
  int get_month();
  int get_year();

  void set_day(int day);
  void set_month(int month);
  void set_year(int year);

  bool operator<(const Date& d);
  bool operator>(const Date& d);
  bool operator==(const Date& d);
  bool operator!=(const Date& d);

private:
  int year;
  int month;
  int day;
};

std::ostream& operator<<(std::ostream& os, const Date& d);

}
