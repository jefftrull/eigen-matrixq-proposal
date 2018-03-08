# Build control file for Eigen sparse Q generation experiments
#
# Copyright (C) 2017 Jeffrey E. Trull <edaskel@att.net>
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required( VERSION 3.6 )

# for packages without a standard CMake find script:
set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# Release build by default
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Default build type (Debug, Release, RelWithDebInfo)" FORCE )
endif()

set( CMAKE_CXX_FLAGS_DEBUG "-g -Og -fno-omit-frame-pointer" )

find_package( Eigen3 )
if( NOT EIGEN3_FOUND )
  message( FATAL_ERROR "could not find Eigen" )
endif()

find_package( Boost 1.65 REQUIRED )

# download and use Google Benchmark
include( ExternalProject )
ExternalProject_Add( GBENCH
  GIT_REPOSITORY https://github.com/google/benchmark
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gbench
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property( GBENCH SOURCE_DIR BINARY_DIR )
add_library( gbench STATIC IMPORTED )

# create include directory, which will be absent at initial run
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${SOURCE_DIR}/include )
set_target_properties( gbench PROPERTIES
  IMPORTED_LOCATION "${BINARY_DIR}/src/libbenchmark.a"
  INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}/include"
  INTERFACE_LINK_LIBRARIES pthread
)
add_dependencies( gbench GBENCH )

# Broadside random matrix verification
add_executable( verify verify.cpp )
set_target_properties( verify PROPERTIES
  CXX_STANDARD 14
)
target_include_directories( verify PUBLIC ${EIGEN3_INCLUDE_DIR} )
target_link_libraries( verify Boost::boost )

# Benchmarking code
add_executable( bench bench.cpp )
set_target_properties( bench PROPERTIES
  CXX_STANDARD 14
)
target_include_directories( bench PUBLIC ${EIGEN3_INCLUDE_DIR} )
target_link_libraries( bench gbench )

