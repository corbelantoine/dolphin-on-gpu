#include "asset.hpp"

namespace fin
{

void Asset::set_closes(std::vector<close> closes)
{
  this->closes = closes;
  this->sort_closes();
}

float Asset::get_return(tm* start_date, tm* end_date)
{
  return 0;
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
