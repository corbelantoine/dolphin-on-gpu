#pragma once

#include "../finance/portfolio.cuh"
#include "../../lib/cvxgen/solver.hpp"

#include <cstdio>
#include <cstdlib>

namespace opt
{

__host__ fin::Portfolio get_optimal_portfolio_gpu(fin::Asset *h_assets, int *map_portfolio_assets,
                                      hlp::Date& d1, hlp::Date& d2,
                                      const int nb_assets, const int nb_p = 10, const int p_size = 20);

__host__ fin::Portfolio get_optimal_portfolio_cpu(fin::Asset *h_assets, int *map_portfolio_assets,
                                    hlp::Date& d1, hlp::Date& d2,
                                    const int nb_assets, const int nb_p = 10, const int p_size = 20);

}
