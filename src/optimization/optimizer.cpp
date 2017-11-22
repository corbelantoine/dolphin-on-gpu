#include "optimizer.hpp"

#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

#include <CGAL/Gmpzf.h>
typedef CGAL::Gmpzf ET;

// program and solution types
typedef CGAL::Quadratic_program<float> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

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
                                    size_t n, size_t nb_p = 10, size_t k = 20,
                                    float min = 0.01, float max = 0.1, float lambda = 1)
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
  optimize_portfolios_kernel<<<DimGrid, DimBlock>>>(d_portfolios, d_sharp
                                                    d1, d2,
                                                    n, nb_p, k,
                                                    min, max, lambda);

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
                                size_t n, size_t nb_p, size_t k,
                                float min, float max, float lambda)
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
    optimize_portfolio(d_portfolios[portfolio_idx], d1, d2, min, max, lambda);
    d_sharp[portfolio_idx] = d_portfolios[portfolio_idx].get_sharp(d1, d2);
  }
}

__device__ void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2,
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
  }
  std::cout<<std::endl;

 p.set_weights(weights);
}

}
