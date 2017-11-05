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

  float get_return(hlp::Date start_date, hlp::Date end_date);

private:
  std::vector<close> closes;

  void sort_closes();
};

}
