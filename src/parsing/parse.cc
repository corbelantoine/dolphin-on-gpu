//
// Created by Antoine Corbel on 05/11/2017.
//

#include <fstream>
#include <printf.h>
#include <iostream>
#include "parse.hh"
#include "../../lib/json.hpp"

using json = nlohmann::json;
using namespace std;

json getData(){
    return NULL;
}


map<int, vector<int>> parse(){
    ifstream inFile;
    string data;
    string x;

    inFile.open("../data/cleaned/135");
    if (!inFile) {
        cout << "Unable to open file";
        exit(1);
    }

    json j;
    inFile >> j;

    map dico;
    //dico.insert(pair<int, vector>(135, new vector()));
    cout << dico;
    return map;
}