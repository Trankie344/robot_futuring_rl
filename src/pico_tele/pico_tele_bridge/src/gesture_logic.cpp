#include "pico_tele_bridge/gesture_logic.hpp"

#include <array>
#include <stdexcept>
#include <utility>

#include "pico_tele_interfaces/msg/button_event.hpp"
#include "pico_tele_interfaces/msg/operator_command.hpp"

namespace pico_tele_bridge {
namespace {

using ButtonEvent = pico_tele_interfaces::msg::ButtonEvent;
using OperatorCommand = pico_tele_interfaces::msg::OperatorCommand;

}  // namespace

GestureDetector::GestureDetector(double hold_duration_s, double input_timeout_s)
    : hold_duration_(hold_duration_s), input_timeout_(input_timeout_s) {
  if (hold_duration_s <= 0.0) {
    throw std::invalid_argument("hold_duration_s must be positive");
  }
  if (input_timeout_s <= 0.0) {
    throw std::invalid_argument("input_timeout_s must be positive");
  }
}

GestureDetector::ButtonSnapshot GestureDetector::snapshot(
    const pico_tele_interfaces::msg::ControllerState & controller) {
  return ButtonSnapshot{
      controller.axis_click,
      controller.primary_button,
      controller.secondary_button,
      controller.menu_button,
  };
}

void GestureDetector::append_transitions(
    const ButtonSnapshot & previous,
    const ButtonSnapshot & current,
    std::uint8_t side,
    std::vector<DigitalButtonEvent> & output) {
  const std::array<std::pair<std::uint8_t, std::pair<bool, bool>>, 4> buttons{{
      {ButtonEvent::BUTTON_AXIS_CLICK, {previous.axis_click, current.axis_click}},
      {ButtonEvent::BUTTON_PRIMARY, {previous.primary, current.primary}},
      {ButtonEvent::BUTTON_SECONDARY, {previous.secondary, current.secondary}},
      {ButtonEvent::BUTTON_MENU, {previous.menu, current.menu}},
  }};
  for (const auto & [button, states] : buttons) {
    if (states.first != states.second) {
      output.push_back(DigitalButtonEvent{side, button, states.second});
    }
  }
}

std::optional<OperatorGesture> GestureDetector::update_chord(
    ChordState & chord,
    bool left_pressed,
    bool right_pressed,
    bool enabled,
    std::uint8_t command,
    const char * gesture,
    TimePoint now) {
  if (!left_pressed && !right_pressed) {
    chord.armed = true;
    chord.timing = false;
    return std::nullopt;
  }
  if (!enabled) {
    chord.armed = false;
    chord.timing = false;
    return std::nullopt;
  }
  if (!left_pressed || !right_pressed || !chord.armed) {
    chord.timing = false;
    return std::nullopt;
  }
  if (!chord.timing) {
    chord.timing = true;
    chord.started_at = now;
    return std::nullopt;
  }

  const auto held = std::chrono::duration<double>(now - chord.started_at);
  if (held < hold_duration_) {
    return std::nullopt;
  }
  chord.armed = false;
  chord.timing = false;
  return OperatorGesture{command, gesture, static_cast<float>(held.count())};
}

GestureOutput GestureDetector::update(
    const pico_tele_interfaces::msg::PicoState & state,
    TimePoint now) {
  poll_timeout(now);
  if (!device_id_.empty() && device_id_ != state.device_id) {
    clear_transient_state();
  }
  device_id_ = state.device_id;
  last_update_ = now;

  GestureOutput output;
  if (!state.left_controller.input_valid || !state.right_controller.input_valid) {
    clear_transient_state();
    last_update_ = now;
    device_id_ = state.device_id;
    return output;
  }

  const auto left = snapshot(state.left_controller);
  const auto right = snapshot(state.right_controller);
  if (have_previous_) {
    append_transitions(previous_left_, left, ButtonEvent::CONTROLLER_LEFT, output.button_events);
    append_transitions(previous_right_, right, ButtonEvent::CONTROLLER_RIGHT, output.button_events);
  } else {
    append_transitions(ButtonSnapshot{}, left, ButtonEvent::CONTROLLER_LEFT, output.button_events);
    append_transitions(ButtonSnapshot{}, right, ButtonEvent::CONTROLLER_RIGHT, output.button_events);
  }
  previous_left_ = left;
  previous_right_ = right;
  have_previous_ = true;

  const bool primary_active = left.primary && right.primary;
  const bool secondary_active = left.secondary && right.secondary;
  const bool conflict = primary_active && secondary_active;
  if (const auto command = update_chord(
          primary_chord_,
          left.primary,
          right.primary,
          !conflict,
          OperatorCommand::COMMAND_RESET,
          "dual_primary_hold",
          now)) {
    output.commands.push_back(*command);
  }
  if (const auto command = update_chord(
          secondary_chord_,
          left.secondary,
          right.secondary,
          !conflict,
          OperatorCommand::COMMAND_TOGGLE_CONTROL_MODE,
          "dual_secondary_hold",
          now)) {
    output.commands.push_back(*command);
  }
  return output;
}

void GestureDetector::poll_timeout(TimePoint now) {
  if (last_update_.has_value() && std::chrono::duration<double>(now - *last_update_) > input_timeout_) {
    clear_transient_state();
    last_update_.reset();
    device_id_.clear();
  }
}

void GestureDetector::clear_transient_state() {
  have_previous_ = false;
  previous_left_ = ButtonSnapshot{};
  previous_right_ = ButtonSnapshot{};
  primary_chord_ = ChordState{};
  secondary_chord_ = ChordState{};
}

void GestureDetector::reset() {
  clear_transient_state();
  last_update_.reset();
  device_id_.clear();
}

}  // namespace pico_tele_bridge
