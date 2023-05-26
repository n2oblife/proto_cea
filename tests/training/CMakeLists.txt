cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project (DeeplyTransformed_LIMA)

find_package(Torch REQUIRED)
find_package(ctranslate2 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (test_jit test_jit.cpp)

target_link_libraries(test_jit "${TORCH_LIBRARIES}")
target_link_libraries(test_jit CTranslate2::ctranslate2)

set_property(TARGET test_jit PROPERTY CXX_STANDARD 14)