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

  // setters
  void set_assets(std::vector<Asset*> assets);
  void set_weights(std::vector<float> weights);

  // getters
  std::vector<Asset*> get_assets() const;
  std::vector<float> get_weights() const;

  // helpers to find optimal portfolio
  std::vector<float> get_returns(hlp::Date start_date, hlp::Date end_date) const;
  std::vector<float> get_covariance(hlp::Date start_date, hlp::Date end_date) const;

  // metrics getters
  float get_return(hlp::Date start_date, hlp::Date end_date) const;
  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;
  float get_sharp(hlp::Date start_date, hlp::Date end_date) const;

  void print_weights() const;

private:
  std::vector<Asset*> assets;
  std::vector<float> weights;
};

}
