
file(GLOB SRCS *.c *.h)

option(BUILD_DLL "build EPAnet2 as dynamic link library. If not CLI version will be build" OFF)

if (BUILD_DLL)
add_definitions(-DDLL)
add_library(epanet2 SHARED ${SRCS})
else (BUILD_DLL)
add_definitions(-DCLE)
add_executable(epanet-seq ${SRCS})
endif (BUILD_DLL)

target_link_libraries(epanet-seq -lm)

