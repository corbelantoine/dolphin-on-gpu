#pragma once

#include <cstdio>
#include <cstdlib>

#include "asset.hpp"
#include "../helpers/date.hpp"

namespace fin
{
class Portfolio
{
public:
  Portfolio(int size, bool gpu = false) = default;
  ~Portfolio() = default;

  // setters
  void set_assets(Asset** assets);
  void set_weights(float* weights);

  // getters
  Asset** get_assets() const;
  float* get_weights() const;

  // helpers to find optimal portfolio
  float* get_returns(hlp::Date start_date, hlp::Date end_date) const;
  float* get_covariance(hlp::Date start_date, hlp::Date end_date) const;

  // metrics getters
  float get_return(hlp::Date start_date, hlp::Date end_date) const;
  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;
  float get_sharp(hlp::Date start_date, hlp::Date end_date) const;

  void print_weights() const;

private:
  bool gpu;
  int size;
  Asset** assets;
  float weights;
};

}
