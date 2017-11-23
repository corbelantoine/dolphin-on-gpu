#pragma once

#include <map>
#include <vector>
#include "../helpers/date.cuh"
#include "../../lib/json.hpp"
#include "../finance/asset.cuh"

std::vector<fin::Asset> getAssets(hlp::Date& start_date, hlp::Date& end_date);
fin::Asset parse(std::string path);
