//
// Created by Antoine Corbel on 05/11/2017.
//

#ifndef DOLPHIN_PRIVATE_PARSE_H
#define DOLPHIN_PRIVATE_PARSE_H


#include <map>
#include <vector>
#include "../helpers/date.hpp"
#include "../../lib/json.hpp"
using namespace std;

vector<float>* getCloses(hlp::Date start_date, hlp::Date end_date);
vector<pair<hlp::Date, float>*> parse(string path);


#endif //DOLPHIN_PRIVATE_PARSE_H
