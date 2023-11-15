#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <streambuf>
#include <sstream> 
#include <iostream>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/tools/hlo_module_loader.h"

int main(int argc, char** argv) 
{
    std::ifstream t("hlo_comp_proto.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string hlo_code = buffer.str();
    std::string format = "hlo";
    std::cout << hlo_code << "\n";
    return 0;
}