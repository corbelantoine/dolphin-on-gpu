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
                fin::Asset asset = parse(std::string("data/cleaned/").append(ent->d_name));
                int size = 0;
                fin::Close* closes = asset.get_closes(&size);
                if (size != 0
                  && closes[0].date == start_date
                  && closes[size - 1].date == end_date)
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
    fin::Asset asset;

    // TODO parse id

    inFile.open(path);
    if (!inFile) {
        std::cout << "Unable to open file";
        return asset;
    }

    json j;
    inFile >> j;
    
    // setting closes
    std::vector<fin::Close> closes(j.size());
    for (int i = 0; i < j.size(); ++i) {
        // create close i
        fin::Close close = fin::Close();
        // get close date
        std::string str_date = j[i][0].get<std::string>();
        const char* c_str_date = str_date.c_str(); 
        // set close date
        close.date = hlp::Date(c_str_date);
        // set close value
        close.value = j[i][1];
        // add close to asset closes
        closes[i] = close;
    }
    // set asset closes
    asset.set_closes(closes);
    return asset;
}
