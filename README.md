# Doorbell Detector

This simple script uses the recording device on your machine (like a raspberry pi, I use an Orange Pi 3 LTS) to classify audio and send an alert to a webhook URL if certain audio events are detected past a threshold.

Note that this is tailored for Linux, and doesn't run on Apple Silicon out of the box because of the lack of `tflite_support` pre-built binaries. (`mediapipe` is newer but less documented).
