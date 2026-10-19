#include "minisgl/server/args.h"
#include <cstring>
#include <iostream>

namespace minisgl {

CliArgs CliArgs::parse(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
        } else if (arg == "--tp" && i + 1 < argc) {
            args.tp = std::stoi(argv[++i]);
        } else if (arg == "--page-size" && i + 1 < argc) {
            args.page_size = std::stoi(argv[++i]);
        } else if (arg == "--max-prefill-length" && i + 1 < argc) {
            args.max_prefill_length = std::stoi(argv[++i]);
        } else if (arg == "--cache" && i + 1 < argc) {
            args.cache = argv[++i];
        } else if (arg == "--shell") {
            args.shell = true;
        }
    }
    return args;
}

} // namespace minisgl
