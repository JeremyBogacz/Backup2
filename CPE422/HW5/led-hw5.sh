#!/bin/bash

# Created by Jeremy Bogacz on 10/12/2021

# Modified bash script written by Derek Molloy. Added an additional if
# statement to blink the LED a user defined number of times. As well as 
# the original if statements that turn the LED on, turn the LED off, get
# the status of the LED, and flash the LED.

# Absolute path to USR3 LED
LED3_PATH=/sys/class/leds/beaglebone:green:usr3

# Function to remove the trigger
function removeTrigger
{
  echo "none" >> "$LED3_PATH/trigger"
}

echo "Starting the LED Bash Script"

# Print if no aruguments are given
if [ $# -eq 0 ]; then
  echo "There are no arguments. Usage is:"
  echo -e " bashLED Command \n  where command is one of "
  echo -e "   on, off, flash, status, or blink  \n e.g. led-hw5 on \n      led-hw5 blink 3 "
  exit 2
fi

echo "The LED Command that was passed is: $1"

# Turn the LED on
if [ "$1" == "on" ]; then
  echo "Turning the LED on"
  removeTrigger
  echo "1" >> "$LED3_PATH/brightness"

# Turn the LED off
elif [ "$1" == "off" ]; then
  echo "Turning the LED off"
  removeTrigger
  echo "0" >> "$LED3_PATH/brightness"

# Flash the LED
elif [ "$1" == "flash" ]; then
  echo "Flashing the LED"
  removeTrigger
  echo "timer" >> "$LED3_PATH/trigger"
  sleep 1
  echo "100" >> "$LED3_PATH/delay_off"
  echo "100" >> "$LED3_PATH/delay_on"

# Get the status of the LED
elif [ "$1" == "status" ]; then
  cat "$LED3_PATH/trigger";

# Blink the LED n times
elif [ "$1" == "blink" ]; then
  echo "Blinking the LED $2 times"
  n=$2
  echo "1" >> "$LED3_PATH/brightness"
  while [ $n -gt 0 ]
  do
    echo "0" >> "$LED3_PATH/brightness"
    sleep 1
    echo "1" >> "$LED3_PATH/brightness"
    sleep 1
    ((n--))
  done

fi

echo "End of the LED Bash Script"
