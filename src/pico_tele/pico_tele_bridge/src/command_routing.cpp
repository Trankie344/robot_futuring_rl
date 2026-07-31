#include "pico_tele_bridge/command_routing.hpp"

#include "pico_tele_interfaces/msg/operator_command.hpp"

namespace pico_tele_bridge {

CommandRoute classify_operator_command(std::uint8_t command) {
  using Message = pico_tele_interfaces::msg::OperatorCommand;
  switch (command) {
    case Message::COMMAND_RESET:
      return CommandRoute::kReset;
    case Message::COMMAND_TOGGLE_CONTROL_MODE:
      return CommandRoute::kToggleControlMode;
    default:
      return CommandRoute::kNone;
  }
}

}  // namespace pico_tele_bridge
