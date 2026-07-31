cmake_minimum_required(VERSION 3.8)
project(pico_tele_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(PICO_TELE_BUILD_SDK_BRIDGE "Build the proprietary PXREA SDK bridge executable" ON)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(pico_tele_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(pico_tele_core
  src/command_routing.cpp
  src/gesture_logic.cpp
  src/tracking_parser.cpp
)
target_compile_features(pico_tele_core PUBLIC cxx_std_17)
target_include_directories(pico_tele_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(pico_tele_core
  builtin_interfaces
  pico_tele_interfaces
  rclcpp
  std_msgs
)
target_link_libraries(pico_tele_core nlohmann_json::nlohmann_json)

add_executable(pico_gesture_mapper src/gesture_mapper.cpp)
target_compile_features(pico_gesture_mapper PUBLIC cxx_std_17)
target_link_libraries(pico_gesture_mapper pico_tele_core)
ament_target_dependencies(pico_gesture_mapper pico_tele_interfaces rclcpp)

add_executable(pico_command_router src/command_router.cpp)
target_compile_features(pico_command_router PUBLIC cxx_std_17)
ament_target_dependencies(pico_command_router pico_tele_interfaces rclcpp std_msgs)

if(PICO_TELE_BUILD_SDK_BRIDGE)
  set(PICO_ROBOT_SDK_ROOT "$ENV{PICO_ROBOT_SDK_ROOT}" CACHE PATH "Root of the extracted PXREA Robot SDK")
  if(NOT PICO_ROBOT_SDK_ROOT)
    message(FATAL_ERROR "PICO_ROBOT_SDK_ROOT is required when PICO_TELE_BUILD_SDK_BRIDGE=ON")
  endif()

  find_path(PICO_ROBOT_SDK_INCLUDE
    NAMES PXREARobotSDK.h
    PATHS "${PICO_ROBOT_SDK_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
  )

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(PICO_ROBOT_SDK_LIBRARY_DIR "${PICO_ROBOT_SDK_ROOT}/linux_aarch64/64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(PICO_ROBOT_SDK_LIBRARY_DIR "${PICO_ROBOT_SDK_ROOT}/linux/64")
  else()
    message(FATAL_ERROR "Unsupported PXREA SDK architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  find_library(PICO_ROBOT_SDK_LIBRARY
    NAMES PXREARobotSDK
    PATHS "${PICO_ROBOT_SDK_LIBRARY_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )

  add_executable(pico_sdk_bridge src/pico_sdk_bridge.cpp)
  target_compile_features(pico_sdk_bridge PUBLIC cxx_std_17)
  target_include_directories(pico_sdk_bridge PRIVATE "${PICO_ROBOT_SDK_INCLUDE}")
  target_link_libraries(pico_sdk_bridge pico_tele_core "${PICO_ROBOT_SDK_LIBRARY}")
  ament_target_dependencies(pico_sdk_bridge pico_tele_interfaces rclcpp)
  set_target_properties(pico_sdk_bridge PROPERTIES
    BUILD_RPATH "${PICO_ROBOT_SDK_LIBRARY_DIR}"
    INSTALL_RPATH "${PICO_ROBOT_SDK_LIBRARY_DIR}"
  )
  install(TARGETS pico_sdk_bridge DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS
  pico_tele_core
  pico_gesture_mapper
  pico_command_router
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(tracking_parser_test test/tracking_parser_test.cpp)
  if(TARGET tracking_parser_test)
    target_link_libraries(tracking_parser_test pico_tele_core)
    ament_target_dependencies(tracking_parser_test builtin_interfaces pico_tele_interfaces)
  endif()
  ament_add_gtest(gesture_logic_test test/gesture_logic_test.cpp)
  if(TARGET gesture_logic_test)
    target_link_libraries(gesture_logic_test pico_tele_core)
    ament_target_dependencies(gesture_logic_test pico_tele_interfaces)
  endif()
  ament_add_gtest(command_routing_test test/command_routing_test.cpp)
  if(TARGET command_routing_test)
    target_link_libraries(command_routing_test pico_tele_core)
    ament_target_dependencies(command_routing_test pico_tele_interfaces)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(pico_tele_core)
ament_export_dependencies(pico_tele_interfaces rclcpp std_msgs)
ament_package()
