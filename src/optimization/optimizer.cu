#include "optimizer.cuh"


namespace opt
{

__constant__ fin::Asset* all_assets;
__constant__ int* portfolio_assets;

__host__
void check_error(cudaError_t err)
{
  if (err != cudaSuccess)
  {
    std::cerr << cudaGetErrorString(err)
    << "in " << __FILE__
    << "at line " << __LINE__
    << std::endl;
    exit(EXIT_FAILURE);
  }
}

__host__ __device__ 
void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2, int verbose)
{
  Workspace work;
  Settings settings;
  Params params;
  Vars vars;

  set_defaults(settings);
  setup_indexing(work, vars);

  const int p_size = 20;
  if (p.get_size() != p_size) 
  {
      printf("optimize_portfolio error: only portfolios of size %d\n", p_size);
      return;
  }

  // setting the quadratic problem
  // set Sigma to the covariance matrix
  float* cov = p.get_covariance(d1, d2);
  int cov_size = p_size * p_size;
  for (int i = 0; i < cov_size; ++i)
    params.Sigma[i] = cov[i];

  // set Returns to returns
  float* returns = p.get_returns(d1, d2);
  for (int i = 0; i < p_size; ++i)
    params.Returns[i] = returns[i];

  params.lambda[0] = 0.8;

  // Solve problem
  settings.verbose = verbose;
  solve(work, settings, params, vars);

  // get solution (optimal weights)
  float weights[p_size];
  for (int i = 0; i < p_size; ++i)
    weights[i] = vars.Weights[i];

  // set portfolio weights
  p.set_weights(weights);

  // free cov and ret
  delete[] cov;
  delete[] returns;
}

__global__
void optimize_portfolios_kernel(float* d_weights,
                                float* d_sharp,
                                hlp::Date& d1,
                                hlp::Date& d2,
                                const int nb_p,
                                const int p_size)
{
  // get portfolio index
  int portfolio_idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (portfolio_idx < nb_p)
  {
    // create portfolio
    const int size = 20;
    fin::Portfolio p = fin::Portfolio();
  
    // getting assets and weights
    fin::Asset* p_assets[size];
    float p_weights[size];
    for (int j = 0; j < p_size; ++j)
    {
      // portfolio_assets is a global __constant__
      int asset_idx = portfolio_assets[portfolio_idx * p_size + j];
      p_weights[j] = 1. / p_size;
      p_assets[j] = all_assets + asset_idx;
    }

    // set portfolio assets and weights
    p.set_assets(p_assets);
    p.set_weights(p_weights);
    
    // optimize portfolio (get optimal weights)
    optimize_portfolio(p, d1, d2, 0);
    
    // save portfolio weights to shared memory
    for (int j = 0; j < p_size; ++j)
        d_weights[portfolio_idx * p_size + j] = p_weights[j];
    
    // set portfolio sharp for further use
    d_sharp[portfolio_idx] = p.get_sharp(d1, d2);
  }
}

__host__
fin::Portfolio get_optimal_portfolio_gpu(fin::Asset** h_assets,
                                         int *map_portfolio_assets,
                                         hlp::Date& d1,
                                         hlp::Date& d2,
                                         const int nb_assets,
                                         const int nb_p,
                                         const int p_size)
{
  float h_weights[nb_p * p_size];
  float h_sharp[nb_p];
  
  float* d_weights;
  float* d_sharp;

  fin::Portfolio optimal_portfolio(p_size);

  // copy values to cuda constants (cpu to gpu)
  cudaMemcpyToSymbol(all_assets, *h_assets, sizeof(fin::Asset) * nb_assets);
  cudaMemcpyToSymbol(portfolio_assets, map_portfolio_assets, sizeof(int) * nb_p * p_size);

  // cuda malloc device portfolios weights and sharps
  cudaError_t err = cudaMalloc((void **) &d_weights, sizeof(float) * nb_p * p_size);
  check_error(err);
  err = cudaMalloc((void **) &d_sharp, sizeof(float) * nb_p);
  check_error(err);

  // TODO adapt grid and block size
  dim3 DimGrid(((nb_assets - 1) / 256, 1, 1));
  dim3 DimBlock(256, 1, 1);
  
  // call kernel
  optimize_portfolios_kernel<<<DimGrid, DimBlock>>>(d_weights, d_sharp,
                                                    d1, d2, nb_p, p_size);
  
  // copy optimized portfolios and their sharp values from gpu to cpu
  cudaMemcpy(h_weights, d_weights, sizeof(float) * nb_p * p_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sharp, d_sharp, sizeof(float) * nb_p, cudaMemcpyDeviceToHost);
   
  // free cuda memory
  cudaFree(d_weights);
  cudaFree(d_sharp);
 
  // get portfolio with max sharp
  int max_idx = 0;
  float max_sharp = h_sharp[0];
  for (int i = 0; i < nb_p; ++i)
  {
    if (h_sharp[i] > max_sharp)
    {
      max_sharp = h_sharp[i];
      max_idx = i;
    }
  }

  // set optimal portfolio
  float opt_weights[p_size];
  fin::Asset* opt_assets[p_size];
  for (int i = 0; i < p_size; ++i)
  {
      int asset_idx = map_portfolio_assets[max_idx * p_size + i];
      opt_assets[i] = *h_assets + asset_idx;
      opt_weights[i] = h_weights[max_idx * p_size + i];
  }
  optimal_portfolio.set_assets(opt_assets);
  optimal_portfolio.set_weights(opt_weights);

  return optimal_portfolio;
}

__host__ 
fin::Portfolio get_optimal_portfolio_cpu(fin::Asset **h_assets,
                                         int *map_portfolio_assets,
                                         hlp::Date& d1,
                                         hlp::Date& d2,
                                         const int nb_assets,
                                         const int nb_p,
                                         const int p_size)
{
  fin::Portfolio optimal_portfolio(p_size);
  float max_sharp = 0;

  // optimize portfolios and return the one with max sharp
  for (int i = 0; i < nb_p; ++i) {
    // create portfolio
    fin::Portfolio p = fin::Portfolio(p_size);
    
    // declare portfolio assets and weights
    fin::Asset* p_assets[p_size];
    float p_weights[p_size];
    for (int j = 0; j < p_size; ++j) {
      // get portfolio assets and weights
      int asset_idx = map_portfolio_assets[i * p_size + j];
      p_weights[j] = 1. / p_size;
      p_assets[j] = *h_assets + asset_idx;
    }
    
    // set portfolio assets and weights
    p.set_assets(p_assets);
    p.set_weights(p_weights);
    
    // optimize portfolio (get optimal weights)
    optimize_portfolio(p, d1, d2, 0);
    
    // get portfolio with max sharp
    float sharp = p.get_sharp(d1, d2);
    if (sharp >= max_sharp) {
      max_sharp = sharp;
      optimal_portfolio = p;
    }
  }

  return optimal_portfolio;
}

}
