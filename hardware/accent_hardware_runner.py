"""Raspberry Pi button-to-accent-classifier runner.

Press the hardware button to record a short utterance from a USB/system audio
input, then classify the speaker accent with the trained accent model.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASR_ROOT = PROJECT_ROOT / "speech_recognition_project"
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from src.inference.predict import InferenceEngine  # noqa: E402
from src.models.svm_model import SVMModel  # noqa: E402


LED_PIN = 17
BUZZER_PIN = 22
BUTTON_PIN = 14
SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SIZE = 1024

logger = logging.getLogger("accent_hardware_runner")


@contextlib.contextmanager
def suppress_alsa_stderr():
    """Hide noisy ALSA/JACK probing messages printed by PyAudio."""
    if os.environ.get("SHOW_ALSA_WARNINGS"):
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


class GpioController:
    """Small wrapper so GPIO setup and cleanup stay in one place."""

    def __init__(
        self,
        led_pin: int = LED_PIN,
        buzzer_pin: int = BUZZER_PIN,
        button_pin: int = BUTTON_PIN,
    ):
        try:
            import RPi.GPIO as GPIO
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO is required on the Raspberry Pi. "
                "Install it with: sudo apt install python3-rpi.gpio"
            ) from exc

        self.GPIO = GPIO
        self.led_pin = led_pin
        self.buzzer_pin = buzzer_pin
        self.button_pin = button_pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.led(False)
        self.buzzer(False)

    def is_button_pressed(self) -> bool:
        return self.GPIO.input(self.button_pin) == self.GPIO.LOW

    def led(self, enabled: bool) -> None:
        self.GPIO.output(self.led_pin, self.GPIO.HIGH if enabled else self.GPIO.LOW)

    def buzzer(self, enabled: bool) -> None:
        self.GPIO.output(
            self.buzzer_pin, self.GPIO.HIGH if enabled else self.GPIO.LOW
        )

    def beep(self, count: int = 1, duration: float = 0.12, frequency: int = 2000) -> None:
        for _ in range(count):
            pwm = None
            try:
                pwm = self.GPIO.PWM(self.buzzer_pin, frequency)
                pwm.start(50)
                time.sleep(duration)
            except Exception:
                self.buzzer(True)
                time.sleep(duration)
            finally:
                if pwm is not None:
                    pwm.stop()
                self.buzzer(False)
            time.sleep(duration)

    def cleanup(self) -> None:
        self.led(False)
        self.buzzer(False)
        self.GPIO.cleanup()


def load_accent_engine(
    model_path: Path,
    scaler_path: Path,
    encoder_path: Path,
) -> InferenceEngine:
    """Load the saved accent-classifier artifacts."""
    missing = [path for path in (model_path, scaler_path, encoder_path) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Accent classifier artifacts are missing. Train the accent model first:\n"
            "  cd speech_recognition_project\n"
            "  python main.py train --target accent --model svm "
            "--data-dir data/raw --metadata data/metadata.csv --save-dir models\n\n"
            f"Missing files:\n{missing_text}"
        )

    model = SVMModel().load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return InferenceEngine(model=model, scaler=scaler, label_encoder=label_encoder)


def _resample_to_target(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample recorded audio to the model's expected 16 kHz sample rate."""
    if source_rate == SAMPLE_RATE:
        return audio

    try:
        import librosa

        return librosa.resample(
            y=audio,
            orig_sr=source_rate,
            target_sr=SAMPLE_RATE,
        ).astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(source_rate, SAMPLE_RATE)
        up = SAMPLE_RATE // divisor
        down = source_rate // divisor
        return resample_poly(audio, up, down).astype(np.float32)


def record_audio(duration: float, device_index: int | None = None) -> np.ndarray:
    """Record mono audio from PyAudio and return 16 kHz float32 samples."""
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is required to record waveform audio. Install it with:\n"
            "  sudo apt install portaudio19-dev python3-pyaudio\n"
            "or: pip install pyaudio"
        ) from exc

    with suppress_alsa_stderr():
        pa = pyaudio.PyAudio()
    stream = None
    try:
        with suppress_alsa_stderr():
            if device_index is None:
                device_info = pa.get_default_input_device_info()
            else:
                device_info = pa.get_device_info_by_index(device_index)
        source_rate = int(float(device_info.get("defaultSampleRate", SAMPLE_RATE)))

        with suppress_alsa_stderr():
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=source_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
            )

        frames: list[bytes] = []
        chunks = int(source_rate / CHUNK_SIZE * duration)
        for _ in range(chunks):
            frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))

        raw = b"".join(frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio = samples / 32768.0
        return _resample_to_target(audio, source_rate)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


def list_audio_input_devices() -> None:
    """Print available PyAudio input devices and exit."""
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is required to list microphone devices. Install it with:\n"
            "  sudo apt install portaudio19-dev python3-pyaudio\n"
            "or: pip install pyaudio"
        ) from exc

    with suppress_alsa_stderr():
        pa = pyaudio.PyAudio()
    try:
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"{index}: {info['name']}")
    finally:
        pa.terminate()


def wait_for_press(gpio: GpioController, poll_interval: float = 0.05) -> None:
    while not gpio.is_button_pressed():
        time.sleep(poll_interval)


def wait_for_release(gpio: GpioController, poll_interval: float = 0.05) -> None:
    while gpio.is_button_pressed():
        time.sleep(poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record speech from Raspberry Pi hardware and classify accent."
    )
    default_models = ASR_ROOT / "models"
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_models / "accent_svm_model.joblib",
        help="Path to the trained accent SVM model.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=default_models / "accent_scaler.joblib",
        help="Path to the trained accent scaler.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=default_models / "accent_label_encoder.joblib",
        help="Path to the trained accent label encoder.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.5,
        help="Seconds to record after the button is pressed.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional PyAudio input device index for the USB microphone.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument("--led-pin", type=int, default=LED_PIN)
    parser.add_argument("--buzzer-pin", type=int, default=BUZZER_PIN)
    parser.add_argument("--button-pin", type=int, default=BUTTON_PIN)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()

    if args.list_devices:
        list_audio_input_devices()
        return

    engine = load_accent_engine(args.model_path, args.scaler_path, args.encoder_path)
    gpio = GpioController(
        led_pin=args.led_pin,
        buzzer_pin=args.buzzer_pin,
        button_pin=args.button_pin,
    )

    logger.info(
        "Ready. Press the button, speak into the USB microphone for %.1f seconds, "
        "then wait for the accent result.",
        args.duration,
    )
    try:
        while True:
            wait_for_press(gpio)
            gpio.led(True)
            gpio.beep(count=1, duration=0.08)
            logger.info("Recording...")

            audio = record_audio(duration=args.duration, device_index=args.device_index)
            result = engine._run_pipeline(audio)

            gpio.led(False)
            if result.is_error:
                gpio.beep(count=3, duration=0.08)
                logger.info("Could not classify: %s", result.error)
            else:
                gpio.beep(count=2, duration=0.08)
                logger.info(
                    "Accent: %s (confidence %.2f)",
                    result.predicted_word,
                    result.confidence,
                )
                for label, probability in result.top_k:
                    logger.info("  %-12s %.3f", label, probability)

            wait_for_release(gpio)
            time.sleep(0.25)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        gpio.cleanup()


if __name__ == "__main__":
    main()
