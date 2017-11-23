#include "optimizer.hpp"


namespace opt
{

__constant__ fin::Asset* all_assets;
__constant__ int* portfolio_assets;

void print_cov(float* cov, int n = 20) {
  for (int i = 0; i < n; ++i){
    for (int j = 0; j < n; ++j)
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

__device__ void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2, int verbose)
{
  Workspace work;
  Settings settings;
  Params params;
  Vars vars;

  set_defaults(settings);
  setup_indexing(work, vars);

  int p_size = 20;

  float* cov = p.get_covariance(d1, d2);
  int cov_size = p_size ** p_size;
  for (int i = 0; i < cov_size; ++i)
    params.Sigma[i] = cov[i];

  int n;
  float* returns = p.get_returns(d1, d2, &n);
  for (int i = 0; i < n; ++i)
    params.Returns[i] = returns[i];

  params.lambda[0] = 0.8;

  /* Solve problem instance for the record. */
  settings.verbose = verbose;
  solve(work, settings, params, vars);

  float weights[p_size];
  for (int i = 0; i < p_size; ++i)
    weights[i] = vars.Weights[i];

  p.set_weights(weights);
}

__global__ void optimize_portfolios_kernel(fin::Portfolio* d_portfolios, float* d_sharp,
                                hlp::Date& d1, hlp::Date& d2,
                                int n, int nb_p, int k)
{
  int portfolio_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (portfolio_idx < nb_p) {
    // create portfolio
    Asset* p_assets[k];
    float p_weights[k];
    for (int j = 0; j < k; ++j) {
      int asset_id = portfolio_assets[portfolio_idx * k + j];
      p_weights[j] = 1. / k;
      p_assets[j] = &all_assets[asset_id];
    }
    // set portfolio assets and weights
    d_portfolios[portfolio_idx].set_assets(p_assets);
    d_portfolios[portfolio_idx].set_weights(p_weights);
    // optimize portfolio (get optimal weights)
    optimize_portfolio(d_portfolios[portfolio_idx], d1, d2, 0);
    // set portfolio sharp for further use
    d_sharp[portfolio_idx] = d_portfolios[portfolio_idx].get_sharp(d1, d2);
  }
}

__host__ fin::Portfolio get_optimal_portfolio(fin::Asset *h_assets, int *port_assets,
                                    hlp::Date& d1, hlp::Date& d2,
                                    int n, int nb_p, int k)
{
  fin::Portfolio h_portfolios[nb_p];
  fin::Portfolio *d_portfolios;

  float h_sharp[nb_p];
  float* d_sharp;

  fin::Portfolio optimal_portfolio;

  // copy values to cuda constants (cpu to gpu)
  cudaMemcpyToSymbol(all_assets, h_assets, sizeof(fin::Asset) * n);
  cudaMemcpyToSymbol(portfolio_assets, port_assets, sizeof(int) * nb_p * k);
  // cuda malloc device portfolios and sharps
  cudaError_t err = cudaMalloc((void **) &d_portfolios, sizeof(fin::Portfolio) * nb_p);
  check_error(err);
  err = cudaMalloc((void **) &d_sharp, sizeof(float) * nb_p);
  check_error(err);

  // TODO adapt grid and block size
  dim3 DimGrid(((n - 1) / 256, 1, 1));
  dim3 DimBlock(256, 1, 1);
  optimize_portfolios_kernel<<<DimGrid, DimBlock>>>(d_portfolios, d_sharp,
                                                    d1, d2,
                                                    n, nb_p, k);

  // copy optimized portfolios and their sharp values from gpu to cpu
  cudaMemcpy(h_portfolios, d_portfolios, sizeof(fin::Portfolio) * nb_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sharp, d_sharp, sizeof(float) * nb_p, cudaMemcpyDeviceToHost);
  // free cuda memory
  cudaFree(d_portfolios);
  cudaFree(d_sharp);

  // get portfolio with max sharp
  float max_sharp = h_sharp[0];
  for (int i = 0; i < nb_p; ++i)
    if (h_sharp[i] > max_sharp) {
      max_sharp = h_sharp[i];
      optimal_portfolio = h_portfolios[i];
    }

  return optimal_portfolio;
}

}
