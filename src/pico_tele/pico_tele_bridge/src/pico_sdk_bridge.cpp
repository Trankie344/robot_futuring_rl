#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "PXREARobotSDK.h"
#include "pico_tele_bridge/tracking_parser.hpp"
#include "pico_tele_interfaces/msg/pico_state.hpp"
#include "rclcpp/rclcpp.hpp"

namespace pico_tele_bridge {
namespace {

template <std::size_t Size>
std::string bounded_string(const char (&value)[Size]) {
  const auto end = std::find(value, value + Size, '\0');
  return std::string(value, end);
}

std::string bounded_c_string(const char * value, std::size_t max_size = 128) {
  if (value == nullptr) {
    return {};
  }
  std::size_t length = 0;
  while (length < max_size && value[length] != '\0') {
    ++length;
  }
  return std::string(value, length);
}

}  // namespace

class PicoSdkBridgeNode final : public rclcpp::Node {
 public:
  PicoSdkBridgeNode() : rclcpp::Node("pico_sdk_bridge") {
    const auto state_topic = declare_parameter<std::string>("state_topic", "/pico_tele/state");
    frame_id_ = declare_parameter<std::string>("frame_id", "pico_tracking_origin");
    configured_device_id_ = declare_parameter<std::string>("device_id", "");
    const auto retry_seconds = declare_parameter<double>("sdk_init_retry_s", 2.0);
    if (retry_seconds <= 0.0) {
      throw std::invalid_argument("sdk_init_retry_s must be positive");
    }

    state_publisher_ = create_publisher<pico_tele_interfaces::msg::PicoState>(
        state_topic, rclcpp::SensorDataQoS());
    if (!configured_device_id_.empty()) {
      selected_device_id_ = configured_device_id_;
      RCLCPP_INFO(get_logger(), "Filtering PICO tracking to device_id=%s", selected_device_id_.c_str());
    }

    if (!initialize_sdk()) {
      retry_timer_ = create_wall_timer(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(retry_seconds)),
          [this]() {
            if (initialize_sdk() && retry_timer_) {
              retry_timer_->cancel();
            }
          });
    }
  }

  ~PicoSdkBridgeNode() override {
    if (retry_timer_) {
      retry_timer_->cancel();
    }
    if (sdk_initialized_) {
      PXREADeinit();
      sdk_initialized_ = false;
    }
  }

  static void sdk_callback(
      void * context,
      PXREAClientCallbackType type,
      int status,
      void * user_data) {
    if (context == nullptr) {
      return;
    }
    static_cast<PicoSdkBridgeNode *>(context)->on_sdk_callback(type, status, user_data);
  }

 private:
  bool initialize_sdk() {
    if (sdk_initialized_) {
      return true;
    }
    const int result = PXREAInit(this, &PicoSdkBridgeNode::sdk_callback, static_cast<unsigned>(PXREAFullMask));
    if (result != 0) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "PXREAInit failed with code %d; waiting for PC Service", result);
      return false;
    }
    sdk_initialized_ = true;
    RCLCPP_INFO(get_logger(), "PXREA SDK initialized; waiting for PC Service and PICO tracking data");
    return true;
  }

  bool accept_device(const std::string & device_id) {
    if (device_id.empty()) {
      return false;
    }
    std::lock_guard<std::mutex> lock(device_mutex_);
    if (!configured_device_id_.empty()) {
      return device_id == configured_device_id_;
    }
    if (selected_device_id_.empty()) {
      selected_device_id_ = device_id;
      RCLCPP_INFO(get_logger(), "Locked PICO tracking to first device_id=%s", device_id.c_str());
    }
    return device_id == selected_device_id_;
  }

  void release_device_if_selected(const std::string & device_id) {
    if (!configured_device_id_.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(device_mutex_);
    if (selected_device_id_ == device_id) {
      RCLCPP_INFO(get_logger(), "Released missing PICO device_id=%s", device_id.c_str());
      selected_device_id_.clear();
    }
  }

  void release_auto_selected_device() {
    if (!configured_device_id_.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(device_mutex_);
    if (!selected_device_id_.empty()) {
      RCLCPP_INFO(
          get_logger(),
          "Released PICO device_id=%s after PC Service disconnect",
          selected_device_id_.c_str());
      selected_device_id_.clear();
    }
  }

  void on_tracking_state(PXREADevStateJson & state_json) {
    const auto device_id = bounded_string(state_json.devID);
    if (!accept_device(device_id)) {
      return;
    }
    const auto envelope = bounded_string(state_json.stateJson);
    const auto receipt_stamp = get_clock()->now().to_msg();
    auto result = parse_tracking_envelope(envelope, device_id, receipt_stamp, frame_id_);
    if (!result.accepted) {
      if (!result.error.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", result.error.c_str());
      }
      return;
    }
    if (!result.warning.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", result.warning.c_str());
    }
    state_publisher_->publish(result.state);
  }

  void on_sdk_callback(PXREAClientCallbackType type, int status, void * user_data) {
    switch (type) {
      case PXREAServerConnect:
        RCLCPP_INFO(get_logger(), "XRoboToolkit PC Service connected");
        return;
      case PXREAServerDisconnect:
        RCLCPP_WARN(get_logger(), "XRoboToolkit PC Service disconnected");
        release_auto_selected_device();
        return;
      case PXREADeviceFind: {
        const auto device_id = bounded_c_string(static_cast<const char *>(user_data));
        RCLCPP_INFO(get_logger(), "PICO device online: %s", device_id.c_str());
        accept_device(device_id);
        return;
      }
      case PXREADeviceMissing: {
        const auto device_id = bounded_c_string(static_cast<const char *>(user_data));
        RCLCPP_WARN(get_logger(), "PICO device offline: %s", device_id.c_str());
        release_device_if_selected(device_id);
        return;
      }
      case PXREADeviceConnect:
        RCLCPP_INFO(
            get_logger(),
            "PICO device connection update: device=%s status=%d",
            bounded_c_string(static_cast<const char *>(user_data)).c_str(),
            status);
        return;
      case PXREADeviceStateJson:
        if (user_data != nullptr) {
          on_tracking_state(*static_cast<PXREADevStateJson *>(user_data));
        }
        return;
      case PXREADeviceCustomMessage:
        RCLCPP_DEBUG(get_logger(), "Ignoring PXREA custom device message");
        return;
      default:
        RCLCPP_DEBUG(get_logger(), "Ignoring PXREA callback type=%u", static_cast<unsigned>(type));
    }
  }

  std::string frame_id_;
  std::string configured_device_id_;
  std::string selected_device_id_;
  std::mutex device_mutex_;
  bool sdk_initialized_ = false;
  rclcpp::Publisher<pico_tele_interfaces::msg::PicoState>::SharedPtr state_publisher_;
  rclcpp::TimerBase::SharedPtr retry_timer_;
};

}  // namespace pico_tele_bridge

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<pico_tele_bridge::PicoSdkBridgeNode>());
  } catch (const std::exception & exception) {
    RCLCPP_FATAL(rclcpp::get_logger("pico_sdk_bridge"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
