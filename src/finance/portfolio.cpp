#include "portfolio.hpp"

#include <math.h>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace fin
{

// void Portfolio::add_asset(Asset *asset, float shares);

std::vector<std::tuple<Asset*, float>> Portfolio::get_assets() const
{
  return this->assets;
}

void Portfolio::set_assets(std::vector<std::tuple<Asset*, float>> assets)
{
  float sum = std::accumulate(assets.begin(), assets.end(), 0.0,
                              [](float s, std::tuple<Asset*, float> a) -> float {
                                return s + std::get<1>(a);
                              });

  std::cout << sum << std::endl;
  float eps = 0.01;
  if (sum >= 1 - eps && sum <= 1 + eps)
    this->assets = assets;
  else
    throw std::invalid_argument("Weights must sum to 1");
}

float Portfolio::get_return(hlp::Date start_date, hlp::Date end_date) const
{
  std::vector<float> returns(this->assets.size());
  std::transform(this->assets.begin(), this->assets.end(), returns.begin(),
                [&start_date, &end_date](std::tuple<Asset*, float> a) -> float {
                  float ret = std::get<0>(a)->get_return(start_date, end_date);
                  float w = std::get<1>(a);
                  return w * ret;
                });
  return std::accumulate(returns.begin(), returns.end(), 0.0);
}

float Portfolio::get_volatility(hlp::Date start_date, hlp::Date end_date) const
{
  float vol = 0;
  std::vector<float> cov = this->get_covariance(start_date, end_date);
  for (std::size_t i = 0; i != this->assets.size(); ++i)
    for (std::size_t j = 0; j != this->assets.size(); ++j)
    {
      float wi = std::get<1>(this->assets[i]);
      float wj = std::get<1>(this->assets[j]);
      vol += wi * wj * cov[i * this->assets.size() + j];
    }
  return sqrtf(vol);
}

std::vector<float> Portfolio::get_covariance(hlp::Date start_date, hlp::Date end_date) const
{
  std::vector<float> covariance(this->assets.size() * this->assets.size());

  for (std::size_t i = 0; i != this->assets.size(); ++i)
    for (std::size_t j = 0; j != this->assets.size(); ++j)
    {
      std::vector<float> ri = std::get<0>(this->assets[i])->get_returns(start_date, end_date);
      std::vector<float> rj = std::get<0>(this->assets[j])->get_returns(start_date, end_date);
      float ri_avg = std::accumulate(ri.begin(), ri.end(), 0.0) / ri.size();
      float rj_avg = std::accumulate(rj.begin(), rj.end(), 0.0) / rj.size();

      std::size_t ind = i * this->assets.size() + j;
      covariance[ind] = 0;
      for (std::size_t k = 0; k != ri.size(); ++k)
        covariance[ind] += (ri[k] - ri_avg) * (rj[k] - rj_avg);
      covariance[ind] /= ri.size();
    }
  return covariance;
}


}
