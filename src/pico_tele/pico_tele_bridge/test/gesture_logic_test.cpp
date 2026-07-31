#include <chrono>

#include <gtest/gtest.h>

#include "pico_tele_bridge/gesture_logic.hpp"
#include "pico_tele_interfaces/msg/operator_command.hpp"
#include "pico_tele_interfaces/msg/pico_state.hpp"

namespace {

using Detector = pico_tele_bridge::GestureDetector;

Detector::TimePoint at(double seconds) {
  return Detector::TimePoint{} +
         std::chrono::duration_cast<Detector::Clock::duration>(std::chrono::duration<double>(seconds));
}

pico_tele_interfaces::msg::PicoState state(bool primary, bool secondary) {
  pico_tele_interfaces::msg::PicoState value;
  value.device_id = "PICO-DEMO";
  value.left_controller.input_valid = true;
  value.right_controller.input_valid = true;
  value.left_controller.primary_button = primary;
  value.right_controller.primary_button = primary;
  value.left_controller.secondary_button = secondary;
  value.right_controller.secondary_button = secondary;
  return value;
}

TEST(GestureLogic, EmitsResetOnceAfterOneSecond) {
  Detector detector(1.0, 2.0);
  EXPECT_TRUE(detector.update(state(true, false), at(0.0)).commands.empty());
  EXPECT_TRUE(detector.update(state(true, false), at(0.99)).commands.empty());
  const auto triggered = detector.update(state(true, false), at(1.01));
  ASSERT_EQ(triggered.commands.size(), 1U);
  EXPECT_EQ(
      triggered.commands.front().command,
      pico_tele_interfaces::msg::OperatorCommand::COMMAND_RESET);
  EXPECT_TRUE(detector.update(state(true, false), at(2.0)).commands.empty());
}

TEST(GestureLogic, RequiresBothButtonsReleasedBeforeRearming) {
  Detector detector(1.0, 5.0);
  detector.update(state(true, false), at(0.0));
  ASSERT_EQ(detector.update(state(true, false), at(1.1)).commands.size(), 1U);

  auto mixed = state(false, false);
  mixed.left_controller.primary_button = true;
  detector.update(mixed, at(1.2));
  detector.update(state(true, false), at(1.3));
  EXPECT_TRUE(detector.update(state(true, false), at(2.5)).commands.empty());

  detector.update(state(false, false), at(2.6));
  detector.update(state(true, false), at(2.7));
  EXPECT_EQ(detector.update(state(true, false), at(3.8)).commands.size(), 1U);
}

TEST(GestureLogic, ConflictingChordsDoNotTrigger) {
  Detector detector(1.0, 5.0);
  detector.update(state(true, true), at(0.0));
  EXPECT_TRUE(detector.update(state(true, true), at(2.0)).commands.empty());
}

TEST(GestureLogic, TimeoutCancelsPendingHold) {
  Detector detector(1.0, 0.25);
  detector.update(state(false, true), at(0.0));
  detector.poll_timeout(at(0.3));
  detector.update(state(false, true), at(1.0));
  EXPECT_TRUE(detector.update(state(false, true), at(1.5)).commands.empty());
  const auto triggered = detector.update(state(false, true), at(2.1));
  ASSERT_EQ(triggered.commands.size(), 1U);
  EXPECT_EQ(
      triggered.commands.front().command,
      pico_tele_interfaces::msg::OperatorCommand::COMMAND_TOGGLE_CONTROL_MODE);
}

}  // namespace
