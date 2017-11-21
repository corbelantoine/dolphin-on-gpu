#include <iostream>
#include <algorithm>

#include "../helpers/date.hpp"
#include "../parsing/parse.hpp"
#include "../finance/portfolio.hpp"

#include "../../lib/cvxgen/solver.hpp"


void print_ret_vol(fin::Portfolio& p, hlp::Date& start_date, hlp::Date& end_date)
{
  std::cout << "return: " << p.get_return(start_date, end_date)
            << ", volatility: " << p.get_volatility(start_date, end_date)
            << std::endl;
}

fin::Portfolio get_random_portfolio(std::vector<fin::Asset>& assets, size_t n = 10)
{
  fin::Portfolio p;

  std::vector<std::size_t> tmp(assets.size());
  size_t k = 0;
  std::generate(tmp.begin(), tmp.end(), [&k] () { return k++; });
  std::random_shuffle(tmp.begin(), tmp.end());
  std::vector<std::size_t> indices(tmp.begin(), tmp.begin() + n);

  std::vector<std::tuple<fin::Asset*, float>> p_assets(n);
  for (std::size_t i = 0; i != n; ++i)
  {
    float w = 1. / n;
    std::tuple<fin::Asset*, float> tmp = std::make_tuple(&assets[indices[i]], w);
    p_assets[i] = tmp;
  }
  p.set_assets(p_assets);

  return p;
}

void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2, int verbose = 0)
{
  Workspace work;
  Settings settings;
  Params params;
  Vars vars;

  int num_iters;
  set_defaults(settings);
  setup_indexing(work, vars);

  std::vector<float> cov = p.get_covariance(d1, d2);
  for (std::size_t i = 0; i != cov.size(); ++i)
    params.Sigma[i] = cov[i];

  std::vector<float> returns = p.get_returns(d1, d2);
  for (std::size_t i = 0; i != returns.size(); ++i)
    params.Returns[i] = returns[i];

  params.lambda[0] = 1;

  /* Solve problem instance for the record. */
  settings.verbose = verbose;
  num_iters = solve(work, settings, params, vars);

  std::vector<float> weights(20);
  for (size_t i = 0; i != weights.size(); ++i)
    weights[i] = vars.Weights[i];

  p.set_weights(weights);
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
    p.print_weights();
    print_ret_vol(p, d3, d4);
    optimize_portfolio(p, d3, d4, 1);
    std::cout << "after optimization:\n";
    print_ret_vol(p, d3, d4);
    p.print_weights();
  } catch(const std::exception& e) {
    std::cout << e.what() << std::endl ;
  }

  return 0;
}
