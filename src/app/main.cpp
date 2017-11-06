#include <iostream>

#include "../helpers/date.hpp"
#include "../parsing/parse.hpp"

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

  std::vector<fin::Asset> assets = getAssets(hlp::Date::Date("2008-07-01"), hlp::Date::Date("2016-07-01"));
  //for (int i = 0; i < assets.size(); i++){
  //  auto closes = assets[i].get_closes();
  //  for (int j = 0; j < closes.size(); j++)
  //    std::cout << closes[j].value << endl;
  //}

  cout << assets.size() << endl;

  return 0;
}
