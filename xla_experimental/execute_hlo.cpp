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
    std::ifstream t("hlo_comp_proto.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string hlo_code = buffer.str();
    std::string format = "hlo";
    
    struct PJRT_Program pjrt_program;
    pjrt_program.code = (char*)hlo_code.c_str();
    pjrt_program.code_size = (size_t)hlo_code.size();
    pjrt_program.format = format.c_str();
    pjrt_program.format_size = (size_t)format.size();

    std::cout << "HLO Code:\n\n" << pjrt_program.code << "\n\n";
    std::cout << "Code size: " << pjrt_program.code_size << "\n";
    std::cout << "Format: " << pjrt_program.format << "\n";
    std::cout << "Format size: " << pjrt_program.format_size << "\n";
    return 0;
}