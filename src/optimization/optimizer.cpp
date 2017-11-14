#include "optimizer.hpp"

// namespace opt
// {

void optimize_portfolio(fin::Portfolio& p)
{
  Workspace work;
  Settings settings;
  Params params;
  Vars vars;

  int num_iters;
  set_defaults(settings);
  setup_indexing(work, vars);

  std::vector<float> cov = p.get_covariance();
  for (std::size_t i = 0; i != cov.size(); ++i)
    params.Sigma[i] = cov[i];

  std::vector<float> returns = p.get_returns();
  for (std::size_t i = 0; i != returns.size(); ++i)
    params.Returns[i] = returns[i];

  for (std::size_t i = 0; i != returns.size(); ++i){
    for (std::size_t j = 0; j != returns.size(); ++j)
      std::cout <<cov[i * returns.size() + j] << " ";
    std::cout << std::endl;
  }

  // for (std::size_t i = 0; i != returns.size(); ++i)
  //   for (std::size_t j = 0; j != returns.size(); ++j)
  //     params.Sigma[j * returns.size() + i] = cov[i * returns.size() + j];


  params.lambda[0] = 0.1;


  /* Solve problem instance for the record. */
  settings.verbose = 1;
  num_iters = solve(work, settings, params, vars);

  std::vector<float> weights(20);
  for (size_t i = 0; i != weights.size(); ++i)
    weights[i] = vars.Weights[i];

  p.set_weights(weights);

}

// }
