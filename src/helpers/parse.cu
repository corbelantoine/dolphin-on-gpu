#include <fstream>
#include <iostream>
#include "parse.cuh"
#include <vector>
#include <dirent.h>

using json = nlohmann::json;

std::vector<fin::Asset> getAssets(hlp::Date& start_date, hlp::Date& end_date)
{
    std::cout << "Reading all files, this may take some time ..." << std::endl;
    std::vector<fin::Asset> ret;

    DIR *dir = opendir("data/cleaned");
    if (dir) {
        struct dirent *ent = readdir(dir);
        while (ent) {
            if (std::string(ent->d_name).compare(0, 1 , ".")) {
                auto asset = parse(std::string("data/cleaned/").append(ent->d_name));
                int size = 0;
                auto closes = asset.get_closes(&size);
                if (size != 0
                  && closes[0].date <= start_date
                  && closes[size - 1].date >= end_date)
                    ret.push_back(asset);
            }
            ent = readdir(dir);
        }
        closedir (dir);
    } else
        throw std::invalid_argument("There is no file to read");
    std::cout << "Done! All files are read and parsed." << std::endl;
    return ret;
}


fin::Asset parse(std::string path){
    std::ifstream inFile;
    std::string data;
    fin::Asset asset(-1);
    std::vector<fin::Close> vect;

    // TODO parse id

    inFile.open(path);
    if (!inFile) {
        std::cout << "Unable to open file";
        return asset;
    }

    json j;
    inFile >> j;

    for (int i = 0; i < j.size(); i++) {
        fin::Close p = fin::Close();
        p.date = hlp::Date(j[i][0].get<char*>());
        p.value = j[i][1];
        vect.push_back(p);
    }
    asset.set_closes(vect);
    return asset;
}
