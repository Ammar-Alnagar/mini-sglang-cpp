#include "minisgl/core/types.h"
#include <stdexcept>

namespace minisgl {

static std::shared_ptr<Context> g_context = nullptr;

void set_global_context(std::shared_ptr<Context> ctx) {
    g_context = ctx;
}

std::shared_ptr<Context> get_global_context() {
    if (!g_context) {
        g_context = std::make_shared<Context>();
    }
    return g_context;
}

} // namespace minisgl
