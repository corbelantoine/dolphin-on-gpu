#include <iostream>
#include <random>
#include <algorithm>

#include "../helpers/date.cuh"
#include "../helpers/parse.cuh"
#include "../finance/portfolio.cuh"
#include "../optimization/optimizer.cuh"



void filter_assets(fin::Asset** assets, const int size, const int nb_a)
{
  if (size < nb_a)
  {
    std::cerr << "nb_a must be less or equal than total number of assets a_size\n";
    return;
  }
  printf("all assets: %d, nb_a: %d\n", size, nb_a);
  fin::Asset* filtered_assets = new fin::Asset[nb_a];
  for (int i = 0; i < nb_a; ++i)
    filtered_assets[i] = (*assets)[i];
  delete [] *assets;
  *assets = filtered_assets;
  return;
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
  return ret;
}

int* random_map(const int nb_a, const int nb_p, const int p_size = 20)
{
  if (p_size > nb_a)
  {
    std::cerr << "portfolio size must be less or equal than number of total assets a_size\n";
    return 0;
  }
  // allocate memory for the random map
  int* map = new int[nb_p * p_size];
  // filling the map randomly by line
  for (int i = 0; i < nb_p; ++i)
  {
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

    hlp::Date d1 = hlp::Date("2012-01-01");
    hlp::Date d2 = hlp::Date("2017-06-30");

    const int nb_a = 50;
    const int nb_p = 100;
    const int p_size = 20;
    int n;
    fin::Asset** assets = new fin::Asset*;
    try
    {
        std::cout << "Getting assets...\n";
        *assets = get_assets(d1, d2, &n);
        std::cout << "Getting assets done.\n";
    } catch(const std::exception& e)
    {
        std::cout << e.what() << std::endl ;
    }
    std::cout << "filtering assets...\n";
    filter_assets(assets, n, nb_a);
    std::cout << "Getting random map...\n";
    int* map = random_map(nb_a, nb_p, p_size);
    std::cout << "Getting random map done.\n";
    std::cout << "Finding best portfolio...\n";
    fin::Portfolio p = opt::get_optimal_portfolio_cpu(assets, map, d1, d2, nb_a, nb_p, p_size);
    std::cout << "Finding best portfolio done.\n";

    p.print_weights();
    
    std::cout << "sharp: " << p.get_sharp(d1, d2) << std::endl;
    auto weights = p.get_weights();
    auto passets = p.get_assets();
    for (int i = 0; i < p.get_size(); i++)
        std::cout << "Asset: " << passets[i]->get_id() << " Weight: " << weights[i] << std::endl;


    delete [] *assets;
    delete assets;

    return 0;
}
