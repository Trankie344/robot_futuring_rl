uint8 COMMAND_RESET=1
uint8 COMMAND_TOGGLE_CONTROL_MODE=2

std_msgs/Header header
string device_id
uint8 command
string gesture
float32 held_seconds
