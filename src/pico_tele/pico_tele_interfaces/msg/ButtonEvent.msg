uint8 CONTROLLER_LEFT=1
uint8 CONTROLLER_RIGHT=2

uint8 BUTTON_AXIS_CLICK=1
uint8 BUTTON_PRIMARY=2
uint8 BUTTON_SECONDARY=3
uint8 BUTTON_MENU=4

std_msgs/Header header
string device_id
uint8 controller
uint8 button
bool pressed
float32 value
