#pragma once

#include <string>
#include <string_view>

#include "builtin_interfaces/msg/time.hpp"
#include "pico_tele_interfaces/msg/pico_state.hpp"

namespace pico_tele_bridge {

struct TrackingParseResult {
  bool accepted = false;
  pico_tele_interfaces::msg::PicoState state;
  std::string warning;
  std::string error;
};

TrackingParseResult parse_tracking_envelope(
    std::string_view envelope,
    std::string_view device_id,
    const builtin_interfaces::msg::Time & receipt_stamp,
    std::string_view frame_id);

}  // namespace pico_tele_bridge
