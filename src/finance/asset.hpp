#pragma once

#include <vector>
#include "../helpers/date.hpp"

namespace fin
{

struct close
{
  hlp::Date date;
  float value;
};

class Asset
{
public:
  Asset() = default;
  ~Asset() = default;

  void set_closes(std::vector<close> closes);

  std::vector<close> get_closes() const;
  std::vector<close> get_closes(hlp::Date start_date, hlp::Date end_date) const;

  float get_return(hlp::Date start_date, hlp::Date end_date) const;
  std::vector<float> get_returns(hlp::Date start_date, hlp::Date end_date) const;


  float get_volatility(hlp::Date start_date, hlp::Date end_date) const;

private:
  std::vector<close> closes;

  void sort_closes();
};

}
