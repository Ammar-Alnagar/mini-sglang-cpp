#ifndef MINISGL_SERVER_ARGS_H
#define MINISGL_SERVER_ARGS_H

#include <string>

namespace minisgl {

struct CliArgs {
    std::string model{"Qwen/Qwen3-0.6B"};
    std::string host{"0.0.0.0"};
    int port{1919};
    int tp{1};
    int page_size{1};
    int max_prefill_length{1024};
    std::string cache{"radix"};
    bool shell{false};

    static CliArgs parse(int argc, char** argv);
};

} // namespace minisgl

#endif // MINISGL_SERVER_ARGS_H
