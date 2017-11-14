#pragma once

#include "../finance/portfolio.hpp"

extern "C"
{
#include "../../lib/cvxgen/solver.h"
}

namespace opt
{

void optimize_portfolio(fin::Portfolio& p);

}
