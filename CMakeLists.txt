cmake_minimum_required(VERSION 2.6)
project(EPANet2-pardiso)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS  "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -i8)

add_subdirectory(src)
