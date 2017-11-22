#include <iostream>
#include <algorithm>

#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

#include <CGAL/Gmpzf.h>
typedef CGAL::Gmpzf ET;

// program and solution types
typedef CGAL::Quadratic_program<float> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

#include "../helpers/date.hpp"
#include "../parsing/parse.hpp"
#include "../finance/portfolio.hpp"


void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2,
  float min, float max, float lambda)
{
  /*
  arg1 : set default relation to EQUAL
  arg2 : permit default lower bound
  arg3 : set default lower bound to min
  arg4 : permit default upper bound
  arg5 : set default upper bound to max
  */
  Program qp (CGAL::EQUAL, true, min, true, max);
  int n = p.get_assets().size();

  std::vector<float> returns = p.get_returns(d1, d2);

  for (int j = 0; j < n; ++j) {
    // set A (sum of weights : A = (1, ..., 1))
    qp.set_a(j, 0, 1);
    // set c to - lambda * return
    qp.set_c(j, -lambda * returns[j]);
  }
  // set b to 1 (sum of weights = 1.0)
  qp.set_b(0, 1.0);

  // set D to covrariance matrix
  std::vector<float> cov = p.get_covariance(d1, d2);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j)
      qp.set_d(i, j, cov[i * n + j]);

  // solve the program, using ET as the exact type
  Solution sol = CGAL::solve_quadratic_program(qp, ET());

  // get optimal weights
  Solution::Variable_value_iterator opt_weights = sol.variable_values_begin();
  std::vector<float> weights(n);
  for (int i = 0; i < n; ++i) {
    CGAL::Quotient<ET> weight = *(opt_weights++);
    weights[i] = (float) CGAL::to_double(weight);
    // print optimal weights
    std::cout << weights[i] << "|";
  }
  std::cout<<std::endl;

 p.set_weights(weights);
}


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

int main(int argc, char* argv[])
{
  // hlp::Date d1 = hlp::Date("2008-07-01");
  // hlp::Date d2 = hlp::Date("2016-07-01");

  hlp::Date d3 = hlp::Date("2010-07-01");
  hlp::Date d4 = hlp::Date("2010-08-01");

  try {
    std::vector<fin::Asset> assets = getAssets(d3, d4);
    std::cout << "Getting random portfolio\n";
    fin::Portfolio p = get_random_portfolio(assets, 20);

    std::cout << "before optimization:\n";
    print_ret_vol(p, d3, d4);
    char rep = 'r';
    float min;
    float max;
    float lambda = 1;
    while (rep == 'r') {
      std::cout<< "enter values in that order: min max lambda\n";
      std::cin >> min;
      std::cin >> max;
      std::cin >> lambda;
      optimize_portfolio(p, d3, d4, min, max, lambda);
      std::cout << "after optimization:\n";
      print_ret_vol(p, d3, d4);
      // p.print_weights();
      // std::cout<< "enter r for repeat, any other character to exit\n";
      // std::cin >> rep;
    }

  } catch(const std::exception& e) {
    std::cout << e.what() << std::endl ;
  }

  return 0;
}
