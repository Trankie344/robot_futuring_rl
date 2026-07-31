#include <memory>
#include <string>

#include "pico_tele_bridge/command_routing.hpp"
#include "pico_tele_interfaces/msg/operator_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/empty.hpp"

namespace pico_tele_bridge {

class CommandRouterNode final : public rclcpp::Node {
 public:
  CommandRouterNode() : rclcpp::Node("pico_command_router") {
    const auto command_topic =
        declare_parameter<std::string>("operator_command_topic", "/pico_tele/operator_command");
    const auto reset_topic = declare_parameter<std::string>("reset_topic", "/pico_tele/reset_request");
    const auto toggle_topic = declare_parameter<std::string>("toggle_topic", "/change_ctrl_mode");
    enable_reset_ = declare_parameter<bool>("enable_reset", true);
    enable_toggle_ = declare_parameter<bool>("enable_toggle", true);

    reset_publisher_ = create_publisher<std_msgs::msg::Empty>(
        reset_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    toggle_publisher_ = create_publisher<std_msgs::msg::Bool>(
        toggle_topic, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    command_subscription_ = create_subscription<pico_tele_interfaces::msg::OperatorCommand>(
        command_topic,
        rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
        [this](const pico_tele_interfaces::msg::OperatorCommand::SharedPtr command) { route(*command); });
  }

 private:
  void route(const pico_tele_interfaces::msg::OperatorCommand & command) {
    switch (classify_operator_command(command.command)) {
      case CommandRoute::kReset:
        if (enable_reset_) {
          reset_publisher_->publish(std_msgs::msg::Empty{});
          RCLCPP_INFO(get_logger(), "Published generic reset request from %s", command.device_id.c_str());
        }
        return;
      case CommandRoute::kToggleControlMode:
        if (enable_toggle_) {
          std_msgs::msg::Bool toggle;
          toggle.data = true;
          toggle_publisher_->publish(toggle);
          RCLCPP_INFO(get_logger(), "Published control-mode toggle request from %s", command.device_id.c_str());
        }
        return;
      case CommandRoute::kNone:
        RCLCPP_WARN(
            get_logger(),
            "Ignoring unknown operator command=%u from device=%s",
            static_cast<unsigned int>(command.command),
            command.device_id.c_str());
        return;
    }
  }

  bool enable_reset_ = true;
  bool enable_toggle_ = true;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr reset_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr toggle_publisher_;
  rclcpp::Subscription<pico_tele_interfaces::msg::OperatorCommand>::SharedPtr command_subscription_;
};

}  // namespace pico_tele_bridge

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pico_tele_bridge::CommandRouterNode>());
  rclcpp::shutdown();
  return 0;
}
