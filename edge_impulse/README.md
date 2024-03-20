# Linux packages required

1. PyAudio
    By default debian only has pyaudio@0.2.11, however that doesn't work on Python@3.10 (https://stackoverflow.com/a/70358664)

    The fix is to upgrade pyaudio
    `sudo apt-get install python3-pyaudio`
    (this may not work because the latest version is only 0.2.11)

    Alternatively, install via pip
    `pip3 install -U pyaudio`
    (this may fail because we need portaudio. proceed to below)

    https://stackoverflow.com/a/64823047
    `sudo apt-get install portaudio19-dev`

2. OpenCV

    Opt for `opencv-python-headless` over `opencv-python`.