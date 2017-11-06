#include "portfolio.hpp"

#include <math.h>
#include <numeric>
#include <stdexcept>

namespace fin
{

void Portfolio::add_asset(Asset asset, float shares);

std::vector<(Asset, float)> Portfolio::get_assets() const
{
  return this->assets;
}

void Portfolio::set_assets(std::vector<(Asset, float)> assets)
{
  float sum = std::accumulate(assets.begin(), assets.end(), 0.0,
                              [](float s, (Asset, float) a) -> float {
                                return s + std::get<1>(a);
                              });
  if (sum != 1)
  {
    throw std::invalid_argument("Weights must sum to 1");
  }
  this->assets = assets;
}

float Portfolio::get_return(hlp::Date start_date, hlp::Date end_date) const
{
  std::vector<float> returns;
  std::transform(this->assets.begin(), this->assets.end(), returns.begin(),
                []((Asset, float) a) -> float {
                  float ret = std::get<0>(a).get_return(start_date, end_date);
                  float w = std::get<1>(a);
                  return w * ret;
                });
  return std::accumulate(returns.begin(), returns.end(), 0.0);
}

float Portfolio::get_volatility(hlp::Date start_date, hlp::Date end_date) const;

std::vector<std::vector<float> Portfolio::get_covariance(hlp::Date start_date, hlp::Date end_date) const;


}
