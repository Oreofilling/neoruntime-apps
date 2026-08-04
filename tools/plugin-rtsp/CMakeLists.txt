cmake_minimum_required(VERSION 3.16)
project(plugin-rtsp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HAL headers (for HalPacket, HalCodecType, hal_log)
set(HAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../hal/include")
set(HAL_COMMON_DIR "${CMAKE_SOURCE_DIR}/../../hal/common")

# Camera-daemon headers (for RtspServer)
set(CAMERA_DAEMON_INCLUDE "${CMAKE_SOURCE_DIR}/../../platform/camera-daemon/include")

# Sources
set(SOURCES
    src/main.cpp
    # Reuse RtspServer from camera-daemon
    ${CMAKE_SOURCE_DIR}/../../platform/camera-daemon/src/rtsp_server.cpp
)

# HAL common sources (for hal_log)
file(GLOB HAL_COMMON_SOURCES "${HAL_COMMON_DIR}/*.c")

add_executable(rtsp-plugin ${SOURCES} ${HAL_COMMON_SOURCES})

target_include_directories(rtsp-plugin PRIVATE
    ${HAL_INCLUDE_DIR}
    ${HAL_COMMON_DIR}
    ${CAMERA_DAEMON_INCLUDE}
)

target_link_libraries(rtsp-plugin PRIVATE pthread)

# Install
install(TARGETS rtsp-plugin DESTINATION /opt/aipc/plugins/rtsp/)
