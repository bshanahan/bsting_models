cmake_minimum_required(VERSION 3.9...3.12)

project(emc-lite LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(bout++ REQUIRED)
add_executable(emc-lite emc-lite.cxx)
target_link_libraries(emc-lite PRIVATE bout++::bout++)
