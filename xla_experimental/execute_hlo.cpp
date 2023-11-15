#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <streambuf>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/service/hlo_parser.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"

int main(int argc, char** argv) {
  tsl::port::InitMain("", &argc, &argv);

    // Load HloModule from file.
    std::string hlo_filename = "./hlo_comp.txt";

    std::ifstream in_file;
    in_file.open(hlo_filename);
    std::stringstream str_stream;
    str_stream << in_file.rdbuf();
    std::string hlo_module_str = str_stream.str();
    std::cout << hlo_module_str << "\n";

    std::unique_ptr<xla::HloModule> test_module = 
        xla::ParseAndReturnUnverifiedModule(absl::string_view(hlo_module_str.c_str()));
    const xla::HloModuleProto test_module_proto = test_module->ToProto();
    return 0;
}