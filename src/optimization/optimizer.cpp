#include "optimizer.hpp"

// namespace opt
// {

void print_cov(std::vector<float> cov, int n = 20) {
  for (std::size_t i = 0; i != n; ++i){
    for (std::size_t j = 0; j != n; ++j)
      std::cout <<cov[i * n + j] << " ";
    std::cout << std::endl;
  }
}

void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2, int verbose)
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

  params.lambda[0] = 0.8;


  /* Solve problem instance for the record. */
  settings.verbose = verbose;
  num_iters = solve(work, settings, params, vars);

  std::vector<float> weights(20);
  for (size_t i = 0; i != weights.size(); ++i)
    weights[i] = vars.Weights[i];

  p.set_weights(weights);

}

// }
