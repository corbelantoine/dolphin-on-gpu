#include <iostream>

#include "../helpers/date.hpp"
#include "../parsing/parse.hpp"

int main(int argc, char* argv[])
{
  std::vector<fin::Asset> assets = getAssets(hlp::Date("2008-07-01"), hlp::Date("2016-07-01"));
  //for (int i = 0; i < assets.size(); i++){
  //  auto closes = assets[i].get_closes();
  //  for (int j = 0; j < closes.size(); j++)
  //    std::cout << closes[j].value << endl;
  //}

  std::cout << assets.size() << std::endl;

  return 0;
}
