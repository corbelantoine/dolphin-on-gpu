#pragma once

#include "../finance/portfolio.hpp"
#include "../../lib/cvxgen/solver.hpp"

//
// namespace opt
// {

void optimize_portfolio(fin::Portfolio& p, hlp::Date& d1, hlp::Date& d2, int verbose = 0);

// }
