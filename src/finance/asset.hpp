#pragma once

#include <vector>
#include "../helpers/date.hpp"

namespace fin
{

struct Close
{
  hlp::Date date;
  float value;
};

class Asset
{
public:
  Asset(int id);
  ~Asset() = default;

  void set_closes(std::vector<Close> closes);

  Close* get_closes() const;
  Close* get_closes(hlp::Date start_date, hlp::Date end_date, int *n) const;

  float get_return() const;
  float get_return(hlp::Date start_date, hlp::Date end_date) const;
  float* get_returns(int* n) const;
  float* get_returns(hlp::Date start_date, hlp::Date end_date, int* n) const;
  float get_volatility() const;
  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;

private:
  int id;
  int size;
  Close* closes;

  void sort_closes(std::vector<Close> closes);
};

}
