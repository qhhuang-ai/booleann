cmake_minimum_required(VERSION 3.16)
project(booleann LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

set(PARLAYANN_ROOT "" CACHE PATH "Path to an unmodified ParlayANN checkout")
set(SIEVE_ROOT "" CACHE PATH "Path to an unmodified SIEVE checkout")

add_subdirectory(src/laion)

if(PARLAYANN_ROOT)
  add_subdirectory(src/categorical)
  add_subdirectory(src/ordered)
else()
  message(STATUS "PARLAYANN_ROOT is unset; categorical and ordered engines are skipped")
endif()

if(SIEVE_ROOT)
  add_subdirectory(src/predicate_cube)
else()
  message(STATUS "SIEVE_ROOT is unset; Predicate-Cube executors are skipped")
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME fragment_advisor_self_test
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/src/advisor/fragment_advisor.py --self-test)
  add_test(
    NAME submission_result_check
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/verify_submission_results.py
            ${CMAKE_SOURCE_DIR}/results/submission_results.json)
endif()
