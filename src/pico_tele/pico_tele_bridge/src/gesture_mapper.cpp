#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "pico_tele_bridge/gesture_logic.hpp"
#include "pico_tele_interfaces/msg/button_event.hpp"
#include "pico_tele_interfaces/msg/operator_command.hpp"
#include "pico_tele_interfaces/msg/pico_state.hpp"
#include "rclcpp/rclcpp.hpp"

namespace pico_tele_bridge {

class GestureMapperNode final : public rclcpp::Node {
 public:
  GestureMapperNode()
      : rclcpp::Node("pico_gesture_mapper"),
        detector_(
            declare_parameter<double>("hold_duration_s", 1.0),
            declare_parameter<double>("input_timeout_s", 0.25)) {
    const auto state_topic = declare_parameter<std::string>("state_topic", "/pico_tele/state");
    const auto event_topic = declare_parameter<std::string>("button_event_topic", "/pico_tele/button_event");
    const auto command_topic =
        declare_parameter<std::string>("operator_command_topic", "/pico_tele/operator_command");

    event_publisher_ = create_publisher<pico_tele_interfaces::msg::ButtonEvent>(
        event_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    command_publisher_ = create_publisher<pico_tele_interfaces::msg::OperatorCommand>(
        command_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    state_subscription_ = create_subscription<pico_tele_interfaces::msg::PicoState>(
        state_topic,
        rclcpp::SensorDataQoS(),
        [this](const pico_tele_interfaces::msg::PicoState::SharedPtr state) { on_state(*state); });
    timeout_timer_ = create_wall_timer(
        std::chrono::milliseconds(50),
        [this]() { detector_.poll_timeout(GestureDetector::Clock::now()); });
  }

 private:
  void on_state(const pico_tele_interfaces::msg::PicoState & state) {
    const auto output = detector_.update(state, GestureDetector::Clock::now());
    for (const auto & transition : output.button_events) {
      pico_tele_interfaces::msg::ButtonEvent event;
      event.header = state.header;
      event.device_id = state.device_id;
      event.controller = transition.controller;
      event.button = transition.button;
      event.pressed = transition.pressed;
      event.value = transition.pressed ? 1.0F : 0.0F;
      event_publisher_->publish(event);
    }
    for (const auto & gesture : output.commands) {
      pico_tele_interfaces::msg::OperatorCommand command;
      command.header = state.header;
      command.device_id = state.device_id;
      command.command = gesture.command;
      command.gesture = gesture.gesture;
      command.held_seconds = gesture.held_seconds;
      command_publisher_->publish(command);
      RCLCPP_INFO(
          get_logger(),
          "Published operator command=%u gesture=%s held=%.3fs device=%s",
          static_cast<unsigned int>(command.command),
          command.gesture.c_str(),
          static_cast<double>(command.held_seconds),
          command.device_id.c_str());
    }
  }

  GestureDetector detector_;
  rclcpp::Publisher<pico_tele_interfaces::msg::ButtonEvent>::SharedPtr event_publisher_;
  rclcpp::Publisher<pico_tele_interfaces::msg::OperatorCommand>::SharedPtr command_publisher_;
  rclcpp::Subscription<pico_tele_interfaces::msg::PicoState>::SharedPtr state_subscription_;
  rclcpp::TimerBase::SharedPtr timeout_timer_;
};

}  // namespace pico_tele_bridge

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<pico_tele_bridge::GestureMapperNode>());
  } catch (const std::exception & exception) {
    RCLCPP_FATAL(rclcpp::get_logger("pico_gesture_mapper"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
