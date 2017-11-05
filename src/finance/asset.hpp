#pragma once

#include <vector>
#include "../helpers/date"

namespace fin
{

struct close
{
  Date date;
  float value;
}

class Asset
{
public:
  Asset() = default;
  ~Asset() = default;

  void set_closes(std::vector<close> closes);

  float get_return(tm* start_date, tm* end_date);

private:
  std::vector<close> closes;

  void sort_closes();
}

}
