# Raspberry Pi USB Microphone Accent Classifier

This folder contains the Raspberry Pi hardware runner for the Kiswahili accent
classification prototype. The UART microphone has been removed. The system now
uses a USB microphone to capture real waveform audio for MFCC feature extraction
and accent classification.

## Files

- `main.py` - primary Raspberry Pi entrypoint.
- `accent_hardware_runner.py` - records USB microphone audio when the button is
  pressed, then classifies the speaker accent.
- `collect_accent_samples.py` - records labelled training samples and appends
  rows to the training metadata CSV.
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

## Collect Accent Training Samples

The checked-in sample metadata does not contain the three target labels yet. Use
the USB microphone to collect labelled examples from real speakers before
training.

Record examples like this:

```bash
cd /home/kiswahili-pi/project-srufm
python3 -u hardware/collect_accent_samples.py --accent coastal --speaker-id coastal_001 --word ndiyo --samples 5 --device-index 1
python3 -u hardware/collect_accent_samples.py --accent nairobi --speaker-id nairobi_001 --word ndiyo --samples 5 --device-index 1
python3 -u hardware/collect_accent_samples.py --accent upcountry --speaker-id upcountry_001 --word ndiyo --samples 5 --device-index 1
```

Repeat this for several words and several speakers per accent. For a useful
prototype, collect at least 30-50 recordings per accent. Better results need more
speakers, not just more repeats from one person.

## Train the Accent Model

The runner needs these trained accent-classifier artifacts:

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
python3 main.py train --target accent --model svm --data-dir data/raw --metadata data/accent_metadata.csv --save-dir models
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
