cmake_minimum_required(VERSION 3.10)
project(FYDP_C_Inference C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -pthread -fPIC")

include_directories(include)

# 1. Build shared library for post-processing and Python ctypes binding
add_library(postprocess SHARED
    src/postprocess.c
)
target_link_libraries(postprocess m)

# 2. Build static library for BPU infer engine
add_library(bpu_yolo_static STATIC
    src/bpu_yolo_infer.c
    src/postprocess.c
    src/v4l2_capture.c
)
target_link_libraries(bpu_yolo_static m pthread)

# 3. Build standalone CLI executable
add_executable(bpu_infer_cli
    src/main_infer.c
)
target_link_libraries(bpu_infer_cli bpu_yolo_static m pthread)

# Copy shared library to python_binding directory for ease of testing
add_custom_command(TARGET postprocess POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:postprocess>
    ${CMAKE_CURRENT_SOURCE_DIR}/python_binding/libpostprocess.so
)
