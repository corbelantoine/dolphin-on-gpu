#pragma once

#include <iostream>

namespace hlp
{

class Date
{

public:
  Date(std::string str_date);
  Date() = default;
  ~Date() = default;

  int get_day() const;
  int get_month() const;
  int get_year() const;

  void set_day(int day);
  void set_month(int month);
  void set_year(int year);

  bool operator<(const Date& d) const;
  bool operator>(const Date& d) const;
  bool operator<=(const Date& d) const;
  bool operator>=(const Date& d) const;
  bool operator==(const Date& d) const;
  bool operator!=(const Date& d) const;

private:
  int year;
  int month;
  int day;
};

std::ostream& operator<<(std::ostream& os, const Date& d);

}
