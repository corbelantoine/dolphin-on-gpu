#pragma once

#include "../finance/portfolio.cuh"
#include "../../lib/cvxgen/solver.hpp"

#include <cstdio>
#include <cstdlib>

namespace opt
{

__host__ fin::Portfolio get_optimal_portfolio(fin::Asset *h_assets, int *port_assets,
                                      hlp::Date& d1, hlp::Date& d2,
                                      int n, int nb_p = 10, int k = 20);

}
