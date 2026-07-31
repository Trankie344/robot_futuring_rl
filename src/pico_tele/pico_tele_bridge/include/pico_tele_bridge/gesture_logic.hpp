#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "pico_tele_interfaces/msg/pico_state.hpp"

namespace pico_tele_bridge {

struct DigitalButtonEvent {
  std::uint8_t controller = 0;
  std::uint8_t button = 0;
  bool pressed = false;
};

struct OperatorGesture {
  std::uint8_t command = 0;
  std::string gesture;
  float held_seconds = 0.0F;
};

struct GestureOutput {
  std::vector<DigitalButtonEvent> button_events;
  std::vector<OperatorGesture> commands;
};

class GestureDetector {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  GestureDetector(double hold_duration_s, double input_timeout_s);

  GestureOutput update(const pico_tele_interfaces::msg::PicoState & state, TimePoint now);
  void poll_timeout(TimePoint now);
  void reset();

 private:
  struct ButtonSnapshot {
    bool axis_click = false;
    bool primary = false;
    bool secondary = false;
    bool menu = false;
  };

  struct ChordState {
    bool armed = true;
    bool timing = false;
    TimePoint started_at{};
  };

  static ButtonSnapshot snapshot(const pico_tele_interfaces::msg::ControllerState & controller);
  static void append_transitions(
      const ButtonSnapshot & previous,
      const ButtonSnapshot & current,
      std::uint8_t side,
      std::vector<DigitalButtonEvent> & output);
  std::optional<OperatorGesture> update_chord(
      ChordState & chord,
      bool left_pressed,
      bool right_pressed,
      bool enabled,
      std::uint8_t command,
      const char * gesture,
      TimePoint now);
  void clear_transient_state();

  std::chrono::duration<double> hold_duration_;
  std::chrono::duration<double> input_timeout_;
  std::optional<TimePoint> last_update_;
  std::string device_id_;
  bool have_previous_ = false;
  ButtonSnapshot previous_left_;
  ButtonSnapshot previous_right_;
  ChordState primary_chord_;
  ChordState secondary_chord_;
};

}  // namespace pico_tele_bridge
