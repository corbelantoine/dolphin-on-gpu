//
// Created by Antoine Corbel on 05/11/2017.
//

#include <fstream>
#include <printf.h>
#include <iostream>
#include "parse.h"
#include "../lib/json.hpp"

using json = nlohmann::json;
using namespace std;

json getData(){
    return NULL;
}


void parse(){
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


    for (int i; i < j.size(); i++){
        cout << j[i][1];
        //istringstream(.toString()) >> get_time(&t, "%Y-%m-%d");
        //cout << std::put_time(&t, "%c");
    }
}