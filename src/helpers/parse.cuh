#pragma once

#include <map>
#include <vector>
#include "../helpers/date.cuh"
#include "../../lib/json.hpp"
#include "../finance/asset.cuh"

fin::Asset* get_assets(hlp::Date& start_date, hlp::Date& end_date, int *size);
