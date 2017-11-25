#include <iostream>
#include <algorithm>

#include "../helpers/date.cuh"
#include "../parsing/parse.hpp"
#include "../finance/portfolio.cuh"
#include "../optimization/optimizer.cuh"



void print_ret_vol(fin::Portfolio& p, hlp::Date& start_date, hlp::Date& end_date)
{
  std::cout << "return: " << p.get_return(start_date, end_date)
            << ", volatility: " << p.get_volatility(start_date, end_date)
            << std::endl;
}

fin::Asset* filter_assets(std::vector<fin::Asset>& assets, int nb_a)
{
  if (assets.size() < nb_a) {
    std::cerr << "nb_a must be less or equal than total number of assets a_size\n";
    return 0;
  }

  fin::Asset* filtered_assets = new fin::Asset[nb_a];
  for (int i = 0; i < nb_a; ++i)
    filtered_assets[i] = assets[i];
  return filtered_assets;
}

fin::Asset* get_assets(hlp::Date d1, hlp::Date d2, int* n)
{
  try {
    std::vector<fin::Asset> assets = getAssets(d3, d4);
    return filter_assets(assets, n);
  } catch(const std::exception& e) {
    std::cout << e.what() << std::endl ;
  }
}

std::vector<int> shuffled_vector(const int size)
{
  std::vector<int> ret(size);
  // generate unique numbers from 0 to size
  int k = 0;
  std::generate(ret.begin(), ret.end(), [&k] () { return k++; });
  // get random function to swap
  std::random_device rd;
  std::mt19937 g(rd());
  // shuffle randomly tmp
  std::shuffle(ret.begin(), ret.end(), g);
  return ret
}

int* random_map(const int nb_a, const int nb_p, const int p_size = 20)
{
  if (p_size < nb_a) {
    std::cerr << "portfolio size must be less or equal than number of total assets a_size\n";
    return 0;
  }
  // allocate memory for the random map
  int* map = new int[nb_p * p_size];
  // filling the map randomly by line
  for (int i = 0; i < nb_p; ++i) {
    // get random unique numbers from 0 to nb_a
    std::vector<int> vec = shuffled_vector(nb_a);
    // take first p_size numbers
    for (int j = 0; j < p_size; ++j)
      map[i * p_size + j] = vec[j];
 }
 return map;
}

int main(int argc, char* argv[])
{

  hlp::Date d1 = hlp::Date("2010-07-01");
  hlp::Date d2 = hlp::Date("2010-08-01");

  const int nb_a = 50;
  const int nb_p = 100;
  const int p_size = 20;

  std::cout << "Getting assets...\n";
  fin::Asset* assets = get_assets(d1, d2, &nb_a);
  std::cout << "Getting assets done.\n";
  std::cout << "Getting random map...\n";
  int* map = random_map(nb_a, nb_p, p_size);
  std::cout << "Getting random map done.\n";
  std::cout << "Finding best portfolio...\n";
  fin::Portfolio p = opt::get_optimal_portfolio_cpu(assets, map, d1, d2, nb_a, nb_p, p_size);
  std::cout << "Finding best portfolio done.\n";

  print_ret_vol(p, d1, d2);



  return 0;
}
