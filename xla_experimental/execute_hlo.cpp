#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <streambuf>
#include <sstream> 
#include <iostream>
#include "xla/pjrt/c/pjrt_c_api.h"

int main(int argc, char** argv) 
{
    std::ifstream t("hlo_comp.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::cout << buffer.str() << "\n";
    return 0;
}