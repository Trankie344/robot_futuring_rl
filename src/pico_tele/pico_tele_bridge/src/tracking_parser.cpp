#include "pico_tele_bridge/tracking_parser.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pico_tele_interfaces/msg/controller_state.hpp"
#include "pico_tele_interfaces/msg/tracked_pose.hpp"

namespace pico_tele_bridge {
namespace {

using Json = nlohmann::json;
using ControllerState = pico_tele_interfaces::msg::ControllerState;
using TrackedPose = pico_tele_interfaces::msg::TrackedPose;

void append_issue(std::string & output, const std::string & issue) {
  if (!output.empty()) {
    output += "; ";
  }
  output += issue;
}

bool parse_int32(const Json & value, std::int32_t & output) {
  if (!value.is_number_integer() && !value.is_number_unsigned()) {
    return false;
  }
  try {
    const auto parsed = value.get<std::int64_t>();
    if (parsed < std::numeric_limits<std::int32_t>::min() ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
      return false;
    }
    output = static_cast<std::int32_t>(parsed);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool parse_timestamp(const Json & value, builtin_interfaces::msg::Time & output) {
  if (!value.is_number_integer() && !value.is_number_unsigned()) {
    return false;
  }
  std::uint64_t timestamp_ns = 0;
  try {
    if (value.is_number_unsigned()) {
      timestamp_ns = value.get<std::uint64_t>();
    } else {
      const auto signed_timestamp = value.get<std::int64_t>();
      if (signed_timestamp < 0) {
        return false;
      }
      timestamp_ns = static_cast<std::uint64_t>(signed_timestamp);
    }
  } catch (const std::exception &) {
    return false;
  }

  const auto seconds = timestamp_ns / 1000000000ULL;
  if (seconds > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return false;
  }
  output.sec = static_cast<std::int32_t>(seconds);
  output.nanosec = static_cast<std::uint32_t>(timestamp_ns % 1000000000ULL);
  return true;
}

bool parse_float(const Json & value, float & output) {
  if (!value.is_number()) {
    return false;
  }
  try {
    const auto parsed = value.get<double>();
    if (!std::isfinite(parsed) || std::abs(parsed) > std::numeric_limits<float>::max()) {
      return false;
    }
    output = static_cast<float>(parsed);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool parse_bool(const Json & value, bool & output) {
  if (!value.is_boolean()) {
    return false;
  }
  output = value.get<bool>();
  return true;
}

bool parse_pose_values(const Json & value, std::vector<double> & values) {
  values.clear();
  if (value.is_array()) {
    if (value.size() != 7) {
      return false;
    }
    for (const auto & item : value) {
      if (!item.is_number()) {
        return false;
      }
      const auto parsed = item.get<double>();
      if (!std::isfinite(parsed)) {
        return false;
      }
      values.push_back(parsed);
    }
    return true;
  }

  if (!value.is_string()) {
    return false;
  }
  std::stringstream stream(value.get<std::string>());
  std::string token;
  while (std::getline(stream, token, ',')) {
    try {
      std::size_t consumed = 0;
      const auto parsed = std::stod(token, &consumed);
      while (consumed < token.size() && std::isspace(static_cast<unsigned char>(token[consumed]))) {
        ++consumed;
      }
      if (consumed != token.size() || !std::isfinite(parsed)) {
        return false;
      }
      values.push_back(parsed);
    } catch (const std::exception &) {
      return false;
    }
  }
  return values.size() == 7;
}

bool parse_pose(const Json & value, TrackedPose & output) {
  std::vector<double> pose;
  if (!parse_pose_values(value, pose)) {
    return false;
  }

  const double quaternion_norm = std::sqrt(
      pose[3] * pose[3] + pose[4] * pose[4] + pose[5] * pose[5] + pose[6] * pose[6]);
  if (!std::isfinite(quaternion_norm) || quaternion_norm <= 1e-8) {
    return false;
  }

  output.pose.position.x = pose[0];
  output.pose.position.y = pose[1];
  output.pose.position.z = pose[2];
  output.pose.orientation.x = pose[3] / quaternion_norm;
  output.pose.orientation.y = pose[4] / quaternion_norm;
  output.pose.orientation.z = pose[5] / quaternion_norm;
  output.pose.orientation.w = pose[6] / quaternion_norm;
  output.valid = true;
  return true;
}

void parse_head(const Json & tracking, TrackedPose & output, std::string & warning) {
  output.valid = false;
  output.tracking_status = TrackedPose::STATUS_UNAVAILABLE;
  const auto iterator = tracking.find("Head");
  if (iterator == tracking.end()) {
    return;
  }
  if (!iterator->is_object()) {
    append_issue(warning, "Head must be an object");
    return;
  }

  const auto & head = *iterator;
  const auto pose = head.find("pose");
  const auto status = head.find("status");
  std::int32_t tracking_status = TrackedPose::STATUS_UNAVAILABLE;
  if (pose == head.end() || !parse_pose(*pose, output)) {
    append_issue(warning, "Head.pose must contain seven finite values and a non-zero quaternion");
  }
  if (status == head.end() || !parse_int32(*status, tracking_status)) {
    append_issue(warning, "Head.status must be an int32");
    output.valid = false;
  } else {
    output.tracking_status = tracking_status;
  }
}

void parse_controller(
    const Json & controllers,
    const char * key,
    ControllerState & output,
    std::string & warning) {
  output.tracking.valid = false;
  output.tracking.tracking_status = TrackedPose::STATUS_UNAVAILABLE;
  output.input_valid = false;
  const auto iterator = controllers.find(key);
  if (iterator == controllers.end()) {
    return;
  }
  if (!iterator->is_object()) {
    append_issue(warning, std::string("Controller.") + key + " must be an object");
    return;
  }

  const auto & controller = *iterator;
  const auto pose = controller.find("pose");
  if (pose == controller.end() || !parse_pose(*pose, output.tracking)) {
    append_issue(
        warning,
        std::string("Controller.") + key + ".pose must contain seven finite values and a non-zero quaternion");
  } else {
    output.tracking.tracking_status = TrackedPose::STATUS_UNSPECIFIED;
  }

  const auto axis_x = controller.find("axisX");
  const auto axis_y = controller.find("axisY");
  const auto axis_click = controller.find("axisClick");
  const auto grip = controller.find("grip");
  const auto trigger = controller.find("trigger");
  const auto primary = controller.find("primaryButton");
  const auto secondary = controller.find("secondaryButton");
  const auto menu = controller.find("menuButton");
  const bool inputs_valid =
      axis_x != controller.end() && parse_float(*axis_x, output.axis_x) &&
      axis_y != controller.end() && parse_float(*axis_y, output.axis_y) &&
      axis_click != controller.end() && parse_bool(*axis_click, output.axis_click) &&
      grip != controller.end() && parse_float(*grip, output.grip) &&
      trigger != controller.end() && parse_float(*trigger, output.trigger) &&
      primary != controller.end() && parse_bool(*primary, output.primary_button) &&
      secondary != controller.end() && parse_bool(*secondary, output.secondary_button) &&
      menu != controller.end() && parse_bool(*menu, output.menu_button);
  output.input_valid = inputs_valid;
  if (!inputs_valid) {
    append_issue(warning, std::string("Controller.") + key + " has invalid or missing input fields");
  }
}

}  // namespace

TrackingParseResult parse_tracking_envelope(
    std::string_view envelope,
    std::string_view device_id,
    const builtin_interfaces::msg::Time & receipt_stamp,
    std::string_view frame_id) {
  TrackingParseResult result;
  Json outer;
  try {
    outer = Json::parse(envelope.begin(), envelope.end());
  } catch (const std::exception & exception) {
    result.error = std::string("invalid PXREA state JSON: ") + exception.what();
    return result;
  }
  if (!outer.is_object()) {
    result.error = "PXREA state JSON must be an object";
    return result;
  }

  const auto function_name = outer.find("functionName");
  if (function_name == outer.end() || !function_name->is_string()) {
    result.error = "PXREA state JSON is missing string functionName";
    return result;
  }
  if (function_name->get<std::string>() != "Tracking") {
    return result;
  }

  const auto value = outer.find("value");
  if (value == outer.end()) {
    result.error = "Tracking envelope is missing value";
    return result;
  }

  Json tracking;
  try {
    if (value->is_string()) {
      tracking = Json::parse(value->get<std::string>());
    } else if (value->is_object()) {
      tracking = *value;
    } else {
      result.error = "Tracking value must be a JSON string or object";
      return result;
    }
  } catch (const std::exception & exception) {
    result.error = std::string("invalid nested Tracking JSON: ") + exception.what();
    return result;
  }

  const auto timestamp = tracking.find("timeStampNs");
  if (timestamp == tracking.end() || !parse_timestamp(*timestamp, result.state.header.stamp)) {
    result.error = "Tracking.timeStampNs must be a non-negative integer representable as ROS time";
    return result;
  }
  const auto input = tracking.find("Input");
  if (input == tracking.end() || !parse_int32(*input, result.state.input_mode)) {
    result.error = "Tracking.Input must be an int32";
    return result;
  }

  result.state.header.frame_id = std::string(frame_id);
  result.state.receipt_stamp = receipt_stamp;
  result.state.device_id = std::string(device_id);
  parse_head(tracking, result.state.head, result.warning);

  result.state.left_controller.tracking.tracking_status = TrackedPose::STATUS_UNAVAILABLE;
  result.state.right_controller.tracking.tracking_status = TrackedPose::STATUS_UNAVAILABLE;
  const auto controllers = tracking.find("Controller");
  if (controllers == tracking.end()) {
    // Missing controllers are represented by the initialized invalid states.
  } else if (!controllers->is_object()) {
    append_issue(result.warning, "Tracking.Controller must be an object");
  } else {
    parse_controller(*controllers, "left", result.state.left_controller, result.warning);
    parse_controller(*controllers, "right", result.state.right_controller, result.warning);
  }

  result.accepted = true;
  return result;
}

}  // namespace pico_tele_bridge
