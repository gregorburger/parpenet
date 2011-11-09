
file(GLOB SRCS *.c *.h)

option(BUILD_DLL "build EPAnet2 as dynamic link library. If not CLI version will be build" OFF)

#add_definitions(-DMKL_ILP64)

if (NOT EXISTS $ENV{MKLROOT}/include/mkl.h)
   message(STATUS "MKLROOT not")
else (NOT EXISTS $ENV{MKLROOT}/include/mkl.h)
   message(STATUS "found MKLROOT at " $ENV{MKLROOT})
endif (NOT EXISTS $ENV{MKLROOT}/include/mkl.h)

set(MKL_ROOT $ENV{MKLROOT})
message(STATUS "gaga " ${MKL_ROOT})
include_directories(${MKL_ROOT}/include/)
link_directories(${MKL_ROOT}/lib/intel64/)

if (BUILD_DLL)
add_definitions(-DDLL)
add_library(epanet2 SHARED ${SRCS})
else (BUILD_DLL)
add_definitions(-DCLE)
add_executable(epanet2 ${SRCS})
endif (BUILD_DLL)

target_link_libraries(epanet2 -lm -lmkl_rt -lpthread)

