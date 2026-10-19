file(GLOB_RECURSE TEST_SOURCES "unit/*.cpp" "integration/*.cpp" "cuda/*.cpp")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE minisgl_lib nlohmann_json::nlohmann_json httplib::httplib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
