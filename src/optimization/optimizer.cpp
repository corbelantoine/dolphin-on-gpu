#include "optimizer.hpp"

namespace opt
{

void optimize_portfolio(fin::Portfolio& p)
{
  Vars vars;
  Params params;
  Workspace work;
  Settings settings;

  int num_iters;
  set_defaults();
  setup_indexing();

  std::vector<float> cov = p.get_covariance();
  for (std::size_t i = 0; i != cov.size(); ++i)
    params.Sigma[i] = cov[i];

  std::vector<float> returns = p.get_returns();
  for (std::size_t i = 0; i != returns.size(); ++i)
    params.Returns[i] = returns[i];

  params.lambda[0] = 0.5670501635426375;


  /* Solve problem instance for the record. */
  settings.verbose = 1;
  num_iters = solve();

  std::vector<float> weights(20);
  for (size_t i = 0; i != weights.size(); ++i)
    weights[i] = vars.Weights[i];

  p.set_weights(weights);

}

}
