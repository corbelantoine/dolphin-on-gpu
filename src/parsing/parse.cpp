#include <fstream>
#include <iostream>
#include "parse.hpp"


using json = nlohmann::json;
using namespace std;

vector<float>* getCloses(hlp::Date start_date, hlp::Date end_date){
    auto data = parse("data/cleaned/135");

    if (data.size() > 0 && data[0]->first <= start_date && data[data.size() - 1]->first >= end_date){
        vector<float>* vect = new vector<float>();
        for (int i = 0; i < data.size(); i++)
            vect->push_back(data[i]->second);
        return vect;
    }
    return NULL;
}


vector<pair<hlp::Date, float>*> parse(string path){
    ifstream inFile;
    string data;
    vector<pair<hlp::Date, float>*> vect;


    inFile.open(path);
    if (!inFile) {
        cout << "Unable to open file";
        return vect;
    }

    json j;
    inFile >> j;

    for (int i; i < j.size(); i++) {
        pair<hlp::Date, float> *p = new pair<hlp::Date, float>(hlp::Date::Date(j[i][0].get<string>()), j[i][1]);
        vect.push_back(p);
    }

    return vect;
}