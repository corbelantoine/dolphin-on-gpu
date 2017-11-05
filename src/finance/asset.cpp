#include "asset.hpp"

namespace fin
{

void Asset::set_closes(std::vector<close> closes)
{
  this->closes = closes;
  this->sort_closes();
}

float Asset::get_return(hlp::Date start_date, hlp::Date end_date)
{
  float v1, v2;
  for (auto const& close: this->closes) {
    if (close.date == start_date)
      v1 = close.value;
    if (close.date == end_date)
      v2 = close.value;
  }
  return (v2 - v1) / v1;
}

// sort using a custom function object
bool date_less(close a, close b)
{
  return a.date < b.date;
}

void Asset::sort_closes()
{
  std::sort(this->closes.begin(), this->closes.end(), date_less);
}

}
