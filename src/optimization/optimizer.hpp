#pragma once

#include "../finance/portfolio.hpp"
#include "../../lib/cvxgen/solver.hpp"

#include <cstdio>
#include <cstdlib>

namespace opt
{

fin::Portfolio get_optimal_portfolio(fin::Asset *h_assets, int *port_assets,
                                      hlp::Date& d1, hlp::Date& d2,
                                      size_t n, size_t nb_p = 10, size_t k = 20);

}
