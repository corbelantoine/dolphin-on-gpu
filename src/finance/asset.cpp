#include "asset.hpp"

namespace fin
{

void Asset::set_closes(std::vector<close> closes)
{
  this.closes = closes;
  this.sort_closes();
}

float Asset::get_return(tm* start_date, tm* end_date)
{

}

void Asset::sort_closes()
{
  // sort using a custom function object
  bool date_less(close a, close b) const
  {
    return a.date < b.date;
  }

  std::sort(this.closes.begin(), this.closes.end(), date_less);
}

}
