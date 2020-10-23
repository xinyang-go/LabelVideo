cmake_minimum_required(VERSION 3.17)
project(LabelVideo)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE EQUAL Debug)
    add_compile_options(-O0)
else ()
    add_compile_options(-O3)
endif ()

find_package(OpenCV 4 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(LabelVideo main.cpp label.cpp)
target_link_libraries(LabelVideo ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES} fmt::fmt)