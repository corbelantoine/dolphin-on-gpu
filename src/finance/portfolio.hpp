#pragma once

#include <vector>
#include "asset.hpp"
#include "../helpers/date.hpp"

namespace fin
{
class Portfolio
{
public:
  Portfolio() = default;
  ~Portfolio() = default;

  std::vector<std::tuple<Asset*, float>> get_assets() const;
  void set_assets(std::vector<std::tuple<Asset*, float>> assets);
  void set_weights(std::vector<float> weights);

  std::vector<float> get_returns(hlp::Date start_date, hlp::Date end_date) const;

  float get_return(hlp::Date start_date, hlp::Date end_date) const;

  std::vector<float> get_covariance(hlp::Date start_date, hlp::Date end_date) const;

  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;

  float get_sharp(hlp::Date start_date, hlp::Date end_date) const;

private:
  std::vector<std::tuple<Asset*, float>> assets;

};

}
