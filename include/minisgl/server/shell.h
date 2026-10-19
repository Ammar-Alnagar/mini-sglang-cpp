#ifndef MINISGL_SERVER_SHELL_H
#define MINISGL_SERVER_SHELL_H

#include "minisgl/scheduler/scheduler.h"
#include "minisgl/tokenizer/tokenizer.h"
#include <memory>

namespace minisgl {

class InteractiveShell {
  public:
    InteractiveShell(std::shared_ptr<Tokenizer> tokenizer, std::shared_ptr<Scheduler> scheduler);

    void run();

  private:
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<Scheduler> scheduler_;
    std::vector<ChatMessage> history_;
};

} // namespace minisgl

#endif // MINISGL_SERVER_SHELL_H
