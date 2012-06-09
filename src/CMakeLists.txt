
file(GLOB SRCS *.c *.h)

option(BUILD_DLL "build EPAnet2 as dynamic link library. If not CLI version will be build" OFF)

link_directories(${CMAKE_BINARY_DIR})

if (BUILD_DLL)
add_definitions(-DDLL)
add_library(epanet2 SHARED ${SRCS})
else (BUILD_DLL)
add_definitions(-DCLE)
add_executable(epanet2 ${SRCS})
endif (BUILD_DLL)

target_link_libraries(epanet2 -lblas -llapack -lpardiso412-GNU430-X86-64 -lm)

