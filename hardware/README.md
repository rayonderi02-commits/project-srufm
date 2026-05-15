# Raspberry Pi Hardware Controller

This folder contains the Raspberry Pi GPIO/UART controller used with the
Kiswahili speech hardware prototype.

## Files

- `main.py` - reads a push button, enables microphone listening over UART, and
  drives an LED/buzzer indicator.
- `accent_hardware_runner.py` - records real microphone audio when the button is
  pressed and classifies the speaker accent with the trained ASR model.

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

## Sound-Level Demo

```bash
cd /path/to/project-srufm/hardware
python3 -u main.py
```

## Accent Classification Runner

Accent classification needs real waveform audio, such as a USB microphone, I2S
microphone, or ALSA/PyAudio-compatible audio input. A UART sound-level module
does not provide enough audio data to classify accent groups.

Train the accent model first:

```bash
cd /path/to/project-srufm/speech_recognition_project
python main.py train --target accent --model svm --data-dir data/raw --metadata data/metadata.csv --save-dir models
```

The metadata must contain labelled examples for the three target accent groups:
`coastal`, `nairobi`, and `upcountry`. The runner looks for these trained files:

- `speech_recognition_project/models/accent_svm_model.joblib`
- `speech_recognition_project/models/accent_scaler.joblib`
- `speech_recognition_project/models/accent_label_encoder.joblib`

Then run the hardware classifier:

```bash
cd /path/to/project-srufm
python3 -u hardware/accent_hardware_runner.py
```

When the button is pressed:

1. The LED turns on.
2. The Pi records a short audio clip from the microphone.
3. The trained accent classifier predicts one of `coastal`, `nairobi`, or
   `upcountry`.
4. The result is printed in the terminal.

If the Pi has multiple audio inputs, list them with PyAudio and pass the chosen
index:

```bash
python3 - <<'PY'
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) > 0:
        print(i, info["name"])
pa.terminate()
PY

python3 -u hardware/accent_hardware_runner.py --device-index 1
```
