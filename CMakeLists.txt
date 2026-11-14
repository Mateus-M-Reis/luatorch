cmake_minimum_required(VERSION 3.18)
project(luatorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED Torch_DIR)
  set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)
find_package(OpenMP)

message(STATUS "luatorch: using libtorch at ${Torch_DIR}")

add_library(luatorch SHARED csrc/luatorch.cpp)
target_include_directories(luatorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
target_link_libraries(luatorch PRIVATE ${TORCH_LIBRARIES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(luatorch PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(luatorch PROPERTIES
  OUTPUT_NAME "luatorch"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")

# Install the Lua sources and the native shim side by side so that
# `require("torch")` works with a plain LUA_PATH entry:
#   cmake --install build --prefix ~/.local
install(DIRECTORY lua/ DESTINATION share/lua/5.1)
install(TARGETS luatorch LIBRARY DESTINATION share/lua/5.1/torch)
