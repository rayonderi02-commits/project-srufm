# Raspberry Pi Hardware Controller

This folder contains the Raspberry Pi GPIO/UART controller used with the
Kiswahili speech hardware prototype.

## Files

- `main.py` - reads a push button, enables microphone listening over UART, and
  drives an LED/buzzer indicator.

## Current Pin Mapping

The Python script uses BCM GPIO numbering:

| Component | BCM GPIO | Physical Pin |
| --- | ---: | ---: |
| LED | GPIO17 | Pin 11 |
| Buzzer | GPIO22 | Pin 15 |
| Push button | GPIO27 | Pin 13 |
| UART TX | GPIO14 | Pin 8 |
| UART RX | GPIO15 | Pin 10 |

The push button should connect GPIO27 to GND when pressed. The script enables
the internal pull-up resistor, so a pressed button reads LOW.

## UART Note

GPIO14 and GPIO15 are reserved for `/dev/ttyS0`. Do not connect the LED,
buzzer, or push button to physical pins 8 or 10 while the microphone is using
UART.

Power the microphone from 3.3V unless the exact module requires 5V and has safe
3.3V UART logic or a level shifter.

## Run

```bash
cd /path/to/project-srufm/hardware
python3 -u main.py
```
