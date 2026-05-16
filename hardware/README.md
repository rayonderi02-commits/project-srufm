# Raspberry Pi USB Microphone Accent Classifier

This folder contains the Raspberry Pi hardware runner for the Kiswahili accent
classification prototype. The UART microphone has been removed. The system now
uses a USB microphone to capture real waveform audio for MFCC feature extraction
and accent classification.

## Files

- `main.py` - primary Raspberry Pi entrypoint.
- `accent_hardware_runner.py` - records USB microphone audio when the button is
  pressed, then classifies the speaker accent.
- `requirements.txt` - Raspberry Pi hardware/audio dependencies.

## Current Pin Mapping

The Python script uses BCM GPIO numbering:

| Component | BCM GPIO | Physical Pin |
| --- | ---: | ---: |
| LED | GPIO17 | Pin 11 |
| Buzzer | GPIO22 | Pin 15 |
| Push button | GPIO27 | Pin 13 |
| USB microphone | USB port | USB port |

The push button should connect GPIO27 to GND when pressed. The script enables
the internal pull-up resistor, so a pressed button reads LOW.

## Install Dependencies

From the repository root on the Raspberry Pi:

```bash
cd /home/kiswahili-pi/project-srufm
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio python3-rpi.gpio
pip install -r speech_recognition_project/requirements.txt
pip install -r hardware/requirements.txt
```

## Confirm the USB Microphone

List ALSA recording devices:

```bash
arecord -l
```

List PyAudio input devices:

```bash
python3 -u hardware/main.py --list-devices
```

If multiple inputs appear, run the classifier with the USB microphone index:

```bash
python3 -u hardware/main.py --device-index 1
```

## Train the Accent Model

The runner needs trained accent-classifier artifacts:

- `speech_recognition_project/models/accent_svm_model.joblib`
- `speech_recognition_project/models/accent_scaler.joblib`
- `speech_recognition_project/models/accent_label_encoder.joblib`

The training metadata must contain labelled examples for:

- `coastal`
- `nairobi`
- `upcountry`

Train from the speech project folder:

```bash
cd /home/kiswahili-pi/project-srufm/speech_recognition_project
python3 main.py train --target accent --model svm --data-dir data/raw --metadata data/metadata.csv --save-dir models
```

## Run the Integrated Hardware System

```bash
cd /home/kiswahili-pi/project-srufm
python3 -u hardware/main.py
```

When the button is pressed:

1. The LED turns on.
2. The Pi records audio from the USB microphone.
3. The MFCC pipeline extracts speech features.
4. The trained accent model classifies the speaker as `coastal`, `nairobi`, or
   `upcountry`.
5. The result and confidence are printed in the terminal.
6. The buzzer gives feedback after classification.
