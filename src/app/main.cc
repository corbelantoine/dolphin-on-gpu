#include <iostream>
#include "parsing/parse.hh"

int main(int argc, char* argv[])
{
  std::cout << "Hello World!"
            << std::endl << argc << argv;
  parse();
  return 0;
}
