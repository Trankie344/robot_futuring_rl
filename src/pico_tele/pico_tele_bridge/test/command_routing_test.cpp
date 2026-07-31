#include <gtest/gtest.h>

#include "pico_tele_bridge/command_routing.hpp"
#include "pico_tele_interfaces/msg/operator_command.hpp"

namespace {

TEST(CommandRouting, ClassifiesSupportedCommands) {
  using Message = pico_tele_interfaces::msg::OperatorCommand;
  EXPECT_EQ(
      pico_tele_bridge::classify_operator_command(Message::COMMAND_RESET),
      pico_tele_bridge::CommandRoute::kReset);
  EXPECT_EQ(
      pico_tele_bridge::classify_operator_command(Message::COMMAND_TOGGLE_CONTROL_MODE),
      pico_tele_bridge::CommandRoute::kToggleControlMode);
}

TEST(CommandRouting, RejectsUnknownCommand) {
  EXPECT_EQ(
      pico_tele_bridge::classify_operator_command(255),
      pico_tele_bridge::CommandRoute::kNone);
}

}  // namespace
