#ifndef MINISGL_SERVER_SERVER_H
#define MINISGL_SERVER_SERVER_H

#include "minisgl/scheduler/scheduler.h"
#include "minisgl/tokenizer/tokenizer.h"
#include <httplib.h>
#include <memory>
#include <string>

namespace minisgl {

class HttpServer {
  public:
    HttpServer(std::string host, int port, std::shared_ptr<Tokenizer> tokenizer,
               std::shared_ptr<Scheduler> scheduler);

    void start();
    void stop();

  private:
    std::string host_;
    int port_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<Scheduler> scheduler_;
    httplib::Server svr_;

    void setup_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
};

} // namespace minisgl

#endif // MINISGL_SERVER_SERVER_H
