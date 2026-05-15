import RPi.GPIO as GPIO
import serial
import time

# Pin setup
# GPIO14/15 are used by /dev/ttyS0 UART, so keep them free for the mic.
LED = 17      # BCM17, physical pin 11
BUZZER = 22   # BCM22, physical pin 15
BUTTON = 27   # BCM27, physical pin 13

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(LED, GPIO.OUT)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Turn everything off at start
GPIO.output(LED, GPIO.LOW)
GPIO.output(BUZZER, GPIO.LOW)

# UART serial for Elechouse mic
ser = serial.Serial('/dev/ttyS0', baudrate=9600, timeout=1)

THRESHOLD = 20000

print("System ready. Hold button on physical pin 13 / BCM27 to listen...")

try:
    while True:
        button_pressed = GPIO.input(BUTTON) == GPIO.LOW
        print("Button pressed:", button_pressed)

        if button_pressed:
            # LED shows that the microphone/listening mode is active.
            GPIO.output(LED, GPIO.HIGH)
            print("Listening...")

            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print("Raw data:", data)

                sound_level = int.from_bytes(data, 'big') if data else 0
                print("Sound level:", sound_level)

                if sound_level > THRESHOLD:
                    GPIO.output(BUZZER, GPIO.HIGH)
                    print("Sound detected!")
                else:
                    GPIO.output(BUZZER, GPIO.LOW)
                    print("Sound too low.")
            else:
                GPIO.output(BUZZER, GPIO.LOW)
                print("No data from mic.")

        else:
            GPIO.output(LED, GPIO.LOW)
            GPIO.output(BUZZER, GPIO.LOW)

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    print("Cleaning up...")
    ser.close()
    GPIO.cleanup()
    print("Done.")
