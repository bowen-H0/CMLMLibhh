cmake_minimum_required(VERSION 3.10)

project(CMLMLib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(CMLMLibhh STATIC src/Q_learning.cpp)
