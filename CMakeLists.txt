cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(CMLMLib)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 生成静态库
add_library(CMLMLibhh STATIC src/Q_learning.cpp)
