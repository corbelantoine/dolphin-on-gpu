#include <fstream>
#include <iostream>
#include "parse.cuh"
#include <dirent.h>

using json = nlohmann::json;



void parse(std::string path, fin::Asset& asset){
    std::ifstream inFile;
    std::string data;

    // TODO parse id

    inFile.open(path);
    if (!inFile) {
        std::cout << "Unable to open file";
        return;
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
}

bool trim_closes(fin::Asset& asset, hlp::Date& start_date, hlp::Date& end_date) 
{
    int size = 0;
    // get asset closes
    fin::Close* closes = asset.get_closes(&size);
    // check if period is subperiod of all closes 
    if (size != 0 && 
        closes[0].date <= start_date && 
        closes[size - 1].date >= end_date)
    {
        // declare new trimed closes
        std::vector<fin::Close> trimed_closes;
        // fill trimed_closes with closes contained in period
        for (int i = 0; i < size; ++i) 
        {
            if (closes[i].date >= start_date && closes[i].date <= end_date)
            {
                fin::Close close = closes[i];
                trimed_closes.push_back(close); // add close 
            }
        }
        // reset asset closes to closes in start->end date period
        asset.set_closes(trimed_closes); 
        return true;
    } else {
        return false;
    }
}

fin::Asset* get_assets(hlp::Date& start_date, hlp::Date& end_date, int *size)
{
    std::cout << "Reading all files, this may take some time ..." << std::endl;
    fin::Asset* assets = 0;
    int max_size = 1000;
    *size = 0;
    DIR *dir = opendir("data/cleaned");
    if (dir)
    {
        assets = new fin::Asset[max_size];
        struct dirent *ent = readdir(dir);
        fin::Asset asset;
        while (ent) 
        {
            if (std::string(ent->d_name).compare(0, 1 , ".")) 
            {
                parse(std::string("data/cleaned/").append(ent->d_name), asset);
                if (trim_closes(asset, start_date, end_date))
                {
                    assets[*size] = asset;
                    *size += 1;
                    if (*size == max_size) 
                    {
                        fin::Asset *tmp = new fin::Asset [2 * max_size];
                        for (int i = 0; i < max_size; ++i)
                            tmp[i] = assets[i];
                        delete [] assets;
                        assets = tmp;
                        max_size *= 2;
                    }
                }
            }
            ent = readdir(dir);
        }
        closedir (dir);
    } else
        throw std::invalid_argument("There is no file to read");
    fin::Asset *tmp = new fin::Asset [*size];
    for (int i = 0; i < *size; ++i)
        tmp[i] = assets[i];
    delete [] assets;
    assets = tmp;
    std::cout << "Done! All files are read and parsed." << std::endl;
    return assets;
}


