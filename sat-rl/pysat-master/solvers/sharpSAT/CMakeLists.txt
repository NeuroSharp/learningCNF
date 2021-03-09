cmake_minimum_required (VERSION 2.8)
project (sharpSATProject)
set (CMAKE_CXX_STANDARD 11)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3 -DNDEBUG -Wall -fPIC -Wno-deprecated -fno-strict-aliasing")

SET(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB>  -c <TARGET>")
SET(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB>  -c <TARGET>")

# set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -std=c++11 -O3 -g -DNDEBUG -Wall")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall")


IF(UNIX)
  IF(APPLE)
    include_directories(/opt/local/include)
    include_directories(/usr/local/include)
  ELSE(APPLE)
    
    
  ENDIF(APPLE)
ELSE(UNIX)

ENDIF(UNIX)


find_library(GMP_LIB gmp)
find_library(GMPXX_LIB gmpxx)


# include_directories(../)
# include_directories(.)

# add_subdirectory (component_types)

SET(SOURCES alt_component_analyzer.cpp 
#	component_analyzer.cpp
	component_cache.cpp
	component_management.cpp
	instance.cpp
#	new_component_analyzer.cpp
	solver.cpp
	statistics.cpp
	component_types/base_packed_component.cpp 
	component_types/component_archetype.cpp
	)


ADD_LIBRARY(sharpSAT ${SOURCES})# component_types)
# target_link_libraries (sharpSAT /Users/Haddock/Desktop/SAT-ML/Experiments/sat-rl/pysat-master/solvers/sharpSAT/build/component_types/libcomponent_types.a)
target_link_libraries (sharpSAT ${GMP_LIB} ${GMPXX_LIB})
# target_include_directories(sharpSAT PUBLIC ..)



#add_executable (sharpSATRun main.cpp)
#target_link_libraries (sharpSATRun sharpSAT ${GMP_LIB} ${GMPXX_LIB}) 
