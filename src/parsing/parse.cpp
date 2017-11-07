#include <fstream>
#include <iostream>
#include "parse.hpp"
#include <vector>
#include <dirent.h>

using json = nlohmann::json;

std::vector<fin::Asset> getAssets(hlp::Date start_date, hlp::Date end_date){
    std::vector<fin::Asset> ret;

    DIR *dir;
    struct dirent *ent;
    if (dir == opendir("data/cleaned")) {
        while (ent == readdir(dir)) {
            if (std::string(ent->d_name).compare(0, 1 , ".")) {
                auto asset = parse(std::string("data/cleaned/").append(ent->d_name));
                auto closes = asset.get_closes();
                if (!closes.empty()
                  && closes[0].date <= start_date
                  && closes[closes.size() - 1].date >= end_date)
                    ret.push_back(asset);
            }
        }
        closedir (dir);
    } else
        return ret;

    return ret;
}


fin::Asset parse(std::string path){
    std::ifstream inFile;
    std::string data;
    fin::Asset asset(-1);
    std::vector<fin::close> vect;

    // TODO parse id

    inFile.open(path);
    if (!inFile) {
        std::cout << "Unable to open file";
        return asset;
    }

    json j;
    inFile >> j;

    for (int i; i < j.size(); i++) {
        fin::close p = fin::close();
        p.date = hlp::Date(j[i][0].get<std::string>());
        p.value = j[i][1];
        vect.push_back(p);
    }
    asset.set_closes(vect);
    return asset;
}
