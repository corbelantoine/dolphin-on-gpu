#include <iostream>

#include "../helpers/date.hpp"
#include "../parsing/parse.hpp"
#include "../finance/portfolio.hpp"

int main(int argc, char* argv[])
{
  try {
    std::vector<fin::Asset> assets = getAssets(hlp::Date("2008-07-01"), hlp::Date("2016-07-01"));
    for (auto const& asset: assets)
      std::cout << "return: " << asset.get_return()
                << ", volatility: " << asset.get_volatility()
                << std::endl;
  } catch(const std::exception& e) {
    std::cout << e.what() << std::endl ;
  }

  //for (int i = 0; i < assets.size(); i++){
  //  auto closes = assets[i].get_closes();
  //  for (int j = 0; j < closes.size(); j++)
  //    std::cout << closes[j].value << endl;
  //}

  return 0;
}
