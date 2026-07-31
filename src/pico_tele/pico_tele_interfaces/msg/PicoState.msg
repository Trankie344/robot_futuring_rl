# header.stamp is the PICO source timeStampNs converted to ROS time.
# header.frame_id defaults to pico_tracking_origin.
std_msgs/Header header
builtin_interfaces/Time receipt_stamp
string device_id
int32 input_mode
pico_tele_interfaces/TrackedPose head
pico_tele_interfaces/ControllerState left_controller
pico_tele_interfaces/ControllerState right_controller
