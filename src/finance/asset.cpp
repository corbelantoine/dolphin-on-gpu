#include "asset.hpp"

#include <math.h>
#include <numeric>

namespace fin
{

Asset::Asset(int id)
{
  this->id = id;
}

void Asset::set_closes(std::vector<close> closes)
{
  this->closes = closes;
  this->sort_closes();
}

std::vector<close> Asset::get_closes() const
{
  return this->closes;
}

std::vector<close> Asset::get_closes(hlp::Date start_date, hlp::Date end_date) const
{
  int start, end;
  for(std::size_t i = 0; i != this->closes.size(); ++i) {
    if (this->closes[i].date == start_date)
      start = i;
    if (this->closes[i].date == end_date) {
      end = i;
      continue;
    }
  }
  std::vector<close> closes(this->closes.begin() + start, this->closes.begin() + end);
  return closes;
}

float Asset::get_return() const
{
  float v1 = this->closes.front ().value;
  float v2 = this->closes.back().value;
  return (v2 - v1) / v1;
}

float Asset::get_return(hlp::Date start_date, hlp::Date end_date) const
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

std::vector<float> Asset::get_returns() const
{
  std::vector<close> period_closes = this->closes;
  // compute all returns on this period
  std::vector<float> returns (period_closes.size() - 1);
  std::transform(period_closes.begin(), period_closes.end() - 1,
                 period_closes.begin() + 1, returns.begin(),
                 [](close c1, close c2) -> float
                 {
                   float v1 = c1.value;
                   float v2 = c2.value;
                   return (v2 - v1) / v1;
                 });
  return returns;
}

std::vector<float> Asset::get_returns(hlp::Date start_date, hlp::Date end_date) const
{
  std::vector<close> period_closes = this->get_closes(start_date, end_date);
  // compute all returns on this period
  std::vector<float> returns (period_closes.size() - 1);
  std::transform(period_closes.begin(), period_closes.end() - 1,
                 period_closes.begin() + 1, returns.begin(),
                 [](close c1, close c2) -> float
                 {
                   float v1 = c1.value;
                   float v2 = c2.value;
                   return (v2 - v1) / v1;
                 });
  return returns;
}

float Asset::get_volatility() const
{
  std::vector<float> returns = this->get_returns();
  float avg = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

  float vol = 0;
  for(auto const& r : returns){
    vol += pow(r - avg, 2);
  }
  vol /= returns.size();
  vol = sqrtf(vol);

  return vol;
}

float Asset::get_volatility(hlp::Date start_date, hlp::Date end_date) const
{
  std::vector<float> returns = this->get_returns(start_date, end_date);

  float avg = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

  float vol = 0;
  for(auto const& r : returns){
    vol += pow(r - avg, 2);
  }
  vol /= returns.size();
  vol = sqrtf(vol);

  return vol;
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
