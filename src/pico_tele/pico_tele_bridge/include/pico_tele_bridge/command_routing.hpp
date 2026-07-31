#pragma once

#include <cstdint>

namespace pico_tele_bridge {

enum class CommandRoute {
  kNone,
  kReset,
  kToggleControlMode,
};

CommandRoute classify_operator_command(std::uint8_t command);

}  // namespace pico_tele_bridge
