cmake_minimum_required(VERSION 3.22)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(minisgl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Enable CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# Check for CUDA compiler
include(CheckLanguage)
check_language(CUDA)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(MINISGL_USE_CUDA)
    message(STATUS "CUDA support enabled: ${CMAKE_CUDA_COMPILER_VERSION}")
else()
    message(STATUS "CUDA compiler not found or ENABLE_CUDA=OFF. Building in CPU test mode.")
endif()

# Dependencies
include(FetchContent)

# nlohmann_json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# httplib
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Source files
file(GLOB_RECURSE MINISGL_CXX_SOURCES
    "src/core/*.cpp"
    "src/model/*.cpp"
    "src/layer/*.cpp"
    "src/attention/*.cpp"
    "src/cache/*.cpp"
    "src/scheduler/*.cpp"
    "src/distributed/*.cpp"
    "src/tokenizer/*.cpp"
    "src/server/*.cpp"
)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE MINISGL_CUDA_SOURCES
        "kernels/cuda/*.cu"
        "src/kernel/*.cu"
    )
endif()

# Library
add_library(minisgl_lib STATIC ${MINISGL_CXX_SOURCES} ${MINISGL_CUDA_SOURCES})
target_link_libraries(minisgl_lib PRIVATE nlohmann_json::nlohmann_json httplib::httplib)
target_include_directories(minisgl_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set_target_properties(minisgl_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Executable
add_executable(minisgl src/main.cpp)
target_link_libraries(minisgl PRIVATE minisgl_lib nlohmann_json::nlohmann_json httplib::httplib)

# Installation
include(GNUInstallDirs)
install(TARGETS minisgl minisgl_lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/minisgl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
