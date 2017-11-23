#pragma once

#include <cstdio>
#include <cstdlib>

#include "asset.cuh"
#include "../helpers/date.cuh"

namespace fin
{
class Portfolio
{
public:
  Portfolio(int size);
  ~Portfolio();

  // setters
  void set_assets(Asset** assets);
  void set_weights(float* weights);

  // getters
  Asset** get_assets() const;
  float* get_weights() const;
  int get_size() const;

  // helpers to find optimal portfolio
  float* get_returns(hlp::Date start_date, hlp::Date end_date) const;
  float* get_covariance(hlp::Date start_date, hlp::Date end_date) const;

  // metrics getters
  float get_return(hlp::Date start_date, hlp::Date end_date) const;
  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;
  float get_sharp(hlp::Date start_date, hlp::Date end_date) const;

  void print_weights() const;

private:
  int size;
  Asset** assets;
  float* weights;

};

}
