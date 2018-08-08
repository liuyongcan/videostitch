#cmake要求最小版本号
cmake_minimum_required(VERSION 3.2)

#设置环境变量
set(TEST_MATH ${DIR_SRCS})

#指定项目名称
project(VideoStitch)

#头文件目录
include_directories(include)

#源文件目录
aux_source_directory(src DIR_SRCS)

#编译可执行文件
add_executable(${PROJECT_NAME} ${TEST_MATH})

#可执行文件所需要的库
target_link_libraries(${PROJECT_NAME} m)