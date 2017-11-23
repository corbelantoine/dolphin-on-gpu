#include <iostream>
#include <algorithm>

#include "../helpers/date.cuh"
#include "../parsing/parse.hpp"
#include "../finance/portfolio.cuh"
#include "../optimization/optimizer.cuh"

void print_ret_vol(std::vector<fin::Asset>& assets)
{
  for (auto const& asset: assets)
    std::cout << "return: " << asset.get_return()
              << ", volatility: " << asset.get_volatility()
              << std::endl;
}

void print_ret_vol(fin::Portfolio& p, hlp::Date& start_date, hlp::Date& end_date)
{
  std::cout << "return: " << p.get_return(start_date, end_date)
            << ", volatility: " << p.get_volatility(start_date, end_date)
            << std::endl;
}

fin::Portfolio get_random_portfolio(std::vector<fin::Asset>& assets, size_t n = 20)
{
  fin::Portfolio p(n);

  std::vector<std::size_t> tmp(assets.size());
  size_t k = 0;
  std::generate(tmp.begin(), tmp.end(), [&k] () { return k++; });
  std::random_shuffle(tmp.begin(), tmp.end());
  std::vector<std::size_t> indices(tmp.begin(), tmp.begin() + n);

  fin::Asset* p_assets[n];
  float p_weights[n];
  for (int i = 0; i < n; ++i)
  {
    p_assets[i] = &assets[indices[i]];
    p_weights[i] = 1. / n;
  }
  p.set_assets(p_assets);
  p.set_weights(p_weights);

  return p;
}

int main(int argc, char* argv[])
{
  hlp::Date d1 = hlp::Date("2008-07-01");
  hlp::Date d2 = hlp::Date("2016-07-01");

  hlp::Date d3 = hlp::Date("2010-07-01");
  hlp::Date d4 = hlp::Date("2010-08-01");

  try {
    std::vector<fin::Asset> assets = getAssets(d3, d4);
    std::cout << "Getting random portfolio\n";
    fin::Portfolio p = get_random_portfolio(assets, 20);

    std::cout << "before optimization:\n";
    print_ret_vol(p, d3, d4);
//    opt::optimize_portfolio(p, d3, d4);
    std::cout << "after optimization:\n";
    print_ret_vol(p, d3, d4);
  } catch(const std::exception& e) {
    std::cout << e.what() << std::endl ;
  }

  return 0;
}
