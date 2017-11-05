#include <iostream>

#include "../helpers/date.hpp"

int main(int argc, char* argv[])
{
  std::string str1 = "1950-11-23";
  std::string str2 = "1952-10-23";

  hlp::Date d1 = hlp::Date(str1);
  hlp::Date d2 = hlp::Date(str2);

  std::cout << "d1: " << d1 << std::endl
            << "d2: " << d2 << std::endl;

  if (d1 < d2)
    std::cout << "d1 < d2" << std::endl;
  else
    std::cout << "d1 > d2" << std::endl;

  return 0;
}
