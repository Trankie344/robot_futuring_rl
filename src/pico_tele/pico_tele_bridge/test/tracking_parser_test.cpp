#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "builtin_interfaces/msg/time.hpp"
#include "pico_tele_bridge/tracking_parser.hpp"

namespace {

builtin_interfaces::msg::Time receipt_time() {
  builtin_interfaces::msg::Time value;
  value.sec = 20;
  value.nanosec = 30;
  return value;
}

std::string complete_tracking_value() {
  return R"({
    "timeStampNs": 1700000000123456789,
    "Input": 1,
    "Head": {"pose": "1,2,3,0,0,0,1", "status": 3},
    "Controller": {
      "left": {
        "axisX": 0.1, "axisY": -0.2, "axisClick": true,
        "grip": 0.4, "trigger": 0.5,
        "primaryButton": true, "secondaryButton": false, "menuButton": false,
        "pose": "4,5,6,0,0,0,2"
      },
      "right": {
        "axisX": -0.1, "axisY": 0.2, "axisClick": false,
        "grip": 0.6, "trigger": 0.7,
        "primaryButton": false, "secondaryButton": true, "menuButton": false,
        "pose": [7,8,9,0,0,0,1]
      }
    }
  })";
}

TEST(TrackingParser, ParsesCompleteObjectEnvelope) {
  const auto envelope = std::string(R"({"functionName":"Tracking","value":)") +
                        complete_tracking_value() + "}";
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      envelope, "PICO-DEMO", receipt_time(), "pico_tracking_origin");

  ASSERT_TRUE(result.accepted) << result.error;
  EXPECT_TRUE(result.warning.empty()) << result.warning;
  EXPECT_EQ(result.state.device_id, "PICO-DEMO");
  EXPECT_EQ(result.state.header.frame_id, "pico_tracking_origin");
  EXPECT_EQ(result.state.header.stamp.sec, 1700000000);
  EXPECT_EQ(result.state.header.stamp.nanosec, 123456789U);
  EXPECT_TRUE(result.state.head.valid);
  EXPECT_TRUE(result.state.left_controller.input_valid);
  EXPECT_TRUE(result.state.right_controller.input_valid);
  EXPECT_DOUBLE_EQ(result.state.left_controller.tracking.pose.orientation.w, 1.0);
}

TEST(TrackingParser, ParsesCurrentStringWrappedValue) {
  nlohmann::json envelope;
  envelope["functionName"] = "Tracking";
  envelope["value"] = complete_tracking_value();
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      envelope.dump(), "PICO-DEMO", receipt_time(), "pico_tracking_origin");
  EXPECT_TRUE(result.accepted) << result.error;
}

TEST(TrackingParser, IgnoresOtherFunctions) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      R"({"functionName":"Camera","value":{}})",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  EXPECT_FALSE(result.accepted);
  EXPECT_TRUE(result.error.empty());
}

TEST(TrackingParser, RejectsInvalidTimestamp) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      R"({"functionName":"Tracking","value":{"timeStampNs":1.5,"Input":1}})",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  EXPECT_FALSE(result.accepted);
  EXPECT_NE(result.error.find("timeStampNs"), std::string::npos);
}

TEST(TrackingParser, RejectsMalformedOuterJson) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      "{not-json",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  EXPECT_FALSE(result.accepted);
  EXPECT_NE(result.error.find("invalid PXREA state JSON"), std::string::npos);
}

TEST(TrackingParser, MissingControllersRemainInvalid) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      R"({"functionName":"Tracking","value":{"timeStampNs":1,"Input":1}})",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  ASSERT_TRUE(result.accepted) << result.error;
  EXPECT_FALSE(result.state.left_controller.input_valid);
  EXPECT_FALSE(result.state.right_controller.tracking.valid);
  EXPECT_EQ(
      result.state.left_controller.tracking.tracking_status,
      pico_tele_interfaces::msg::TrackedPose::STATUS_UNAVAILABLE);
  EXPECT_EQ(
      result.state.right_controller.tracking.tracking_status,
      pico_tele_interfaces::msg::TrackedPose::STATUS_UNAVAILABLE);
}

TEST(TrackingParser, MalformedControllerContainerRemainsUnavailable) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      R"({"functionName":"Tracking","value":{"timeStampNs":1,"Input":1,"Controller":[]}})",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  ASSERT_TRUE(result.accepted) << result.error;
  EXPECT_FALSE(result.warning.empty());
  EXPECT_EQ(
      result.state.left_controller.tracking.tracking_status,
      pico_tele_interfaces::msg::TrackedPose::STATUS_UNAVAILABLE);
  EXPECT_EQ(
      result.state.right_controller.tracking.tracking_status,
      pico_tele_interfaces::msg::TrackedPose::STATUS_UNAVAILABLE);
}

TEST(TrackingParser, KeepsFrameButInvalidatesMalformedPose) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      R"({"functionName":"Tracking","value":{"timeStampNs":1,"Input":1,"Head":{"pose":"1,2,nan,0,0,0,1","status":3}}})",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  ASSERT_TRUE(result.accepted) << result.error;
  EXPECT_FALSE(result.state.head.valid);
  EXPECT_FALSE(result.warning.empty());
}

TEST(TrackingParser, InvalidControllerPoseDoesNotDiscardValidButtons) {
  const auto result = pico_tele_bridge::parse_tracking_envelope(
      R"({"functionName":"Tracking","value":{"timeStampNs":1,"Input":1,"Controller":{"left":{"axisX":0,"axisY":0,"axisClick":false,"grip":0,"trigger":0,"primaryButton":false,"secondaryButton":false,"menuButton":false,"pose":"1,2,3"}}}})",
      "PICO-DEMO",
      receipt_time(),
      "pico_tracking_origin");
  ASSERT_TRUE(result.accepted) << result.error;
  EXPECT_TRUE(result.state.left_controller.input_valid);
  EXPECT_FALSE(result.state.left_controller.tracking.valid);
}

}  // namespace
