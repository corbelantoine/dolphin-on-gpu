#include "optimizer.hpp"

namespace opt
{

__constant__ fin::Asset* all_assets;
__constant__ int* portfolio_assets;

void print_cov(std::vector<float> cov, int n = 20) {
  for (std::size_t i = 0; i != n; ++i){
    for (std::size_t j = 0; j != n; ++j)
      std::cout <<cov[i * n + j] << " ";
    std::cout << std::endl;
  }
}

__host__ void check_error(cudaError_t err)
{
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err)
    << "in " << __FILE__
    << "at line " << __LINE__
    << std::endl;
    exit(EXIT_FAILURE);
  }
}

__host__ fin::Portfolio get_optimal_portfolio(fin::Asset *h_assets, int *port_assets,
                                    hlp::Date& d1, hlp::Date& d2,
                                    size_t n, size_t nb_p = 10, size_t k = 20)
{
  fin::Portfolio h_portfolios[nb_p];
  fin::Portfolio *d_portfolios;
  
  float h_sharp[nb_p];
  float* d_sharp;

  fin::Portfolio optimal_portfolio;
  
  cudaMemcpyToSymbol(all_assets, h_assets, sizeof(fin::Asset) * n);
  cudaMemcpyToSymbol(portfolio_assets, port_assets, sizeof(int) * nb_p * k);
  cudaError_t err = cudaMalloc((void **) &d_portfolios, sizeof(fin::Portfolio) * nb_p);
  check_error(err);
  cudaError_t err = cudaMalloc((void **) &d_sharp, sizeof(float) * nb_p);
  check_error(err);

  // TODO adapt grid and block size
  dim3 DimGrid(((n - 1) / 256, 1, 1));
  dim3 DimBlock(256, 1, 1);
  optimize_portfolios_kernel<<<DimGrid, DimBlock>>>(d_portfolios, d_sharp,
                                                    d1, d2,
                                                    n, nb_p, k);

  cudaMemcpy(h_portfolios, d_portfolios, sizeof(fin::Portfolio) * nb_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sharp, d_sharp, sizeof(float) * nb_p, cudaMemcpyDeviceToHost);
  cudaFree(d_portfolios);

  // get portfolio with max sharp
  float max_sharp = h_sharp[0];
  for (int i = 0; i < nb_p; ++i)
    if (h_sharp[i] > max_sharp) {
      max_sharp = h_sharp[i];
      optimal_portfolio = h_portfolios[i];
    }

  return optimal_portfolio;
}

__global__ void optimize_portfolios_kernel(fin::Portfolio* d_portfolios, float* d_sharp
                                hlp::Date& d1, hlp::Date& d2,
                                size_t n, size_t nb_p, size_t k)
{
  size_t portfolio_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (portfolio_idx < nb_p) {
    std::vector<std::tuple<fin::Asset*, float>> p_assets(k);
    for (std::size_t j = 0; j != k; ++j)
    {
      float w = 1. / k;
      asset_id = portfolio_assets[portfolio_idx * k + j];
      fin::Asset* asset = &all_assets[asset_id];
      std::tuple<fin::Asset*, float> tmp = std::make_tuple(asset, w);
      p_assets[j] = tmp;
    }
    d_portfolios[portfolio_idx].set_assets(p_assets);
    optimize_portfolio(d_portfolios[portfolio_idx], d1, d2, 0);
    d_sharp[portfolio_idx] = d_portfolios[portfolio_idx].get_sharp(d1, d2);
  }
}

__device__ void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2, int verbose)
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

}
