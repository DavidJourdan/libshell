cmake_minimum_required(VERSION 3.1)
project(libshell)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB LIBFILES src/*.cpp src/SecondFundamentalForm/*.cpp src/MaterialModel/*.cpp)
add_library(libshell STATIC ${LIBFILES})

install(TARGETS libshell DESTINATION lib)

target_include_directories(libshell PRIVATE "/usr/include/eigen3")
target_include_directories(libshell PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
