#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <streambuf>
#include <iostream>
#include "xla/pjrt/c/pjrt_c_api.h"

int main(int argc, char** argv) 
{
    /*
    std::string hlo_filename = "./hlo_comp.txt";
    std::ifstream in_file;
    in_file.open(hlo_filename);
    std::stringstream str_stream;
    str_stream << in_file.rdbuf();
    std::string hlo_module_str = str_stream.str();
    std::cout << hlo_module_str << "\n";
    */
    std::cout << "hello world" << "\n";
    return 0;
}