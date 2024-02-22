## Using Teachable Machine

https://teachablemachine.withgoogle.com/

I did not end up using this because of the lack of accuracy due to inadequate training data. I could not get a wide enough sample of background noise -> a lot of false positives resulting simply from loud unrelated noises.

### Uploading audio samples

Teachable Machine only supports audio samples recorded by it. However, [someone else has reverse-engineered the implementation](https://discuss.tensorflow.org/t/custom-audio-classification-with-teachable-machine/5332/6) to convert our own audio files into the format it expects.

I made an offline web page (`convert_audio_to_teachable_machine_format.html`) that can be used for this purpose. Simply open it in the browser, upload your audio file(s), process it and then download the formatted zip.

It basically uses the WebAudio API to get the raw data from an uploaded audio clip and convert it into a `Float32Array` that Teachable Machine expects.

Customizable parameters are:

- startTime
- endTime
- recordingDuration

I used 4 seconds as my audio clips were 4 seconds long.

Then the JSON format is reconstructed and zipped up so that it can be uploaded to Teachable Machine directly.

### Running inference

1. `pip install -r requirements_teachable_machine.txt`
1. Fill out `.env`. Variable reference below. No quotes necessary.
1. `python3 monitor_with_teachable_machine.py`

| Variable name        | Example value                        | Description                                                                          |
| -------------------- | ------------------------------------ | ------------------------------------------------------------------------------------ |
| SAMPLING_INTERVAL    | 0.5                                  | Runs inference every 0.5 seconds                                                     |
| MODEL_PATH           | soundclassifier_with_metadata.tflite | Path to the downloaded model                                                         |
| MONITORED_CATEGORIES | 1 Intercom                           | Labeled categories (separate by comma for multiple categories) to raise an alert for |
| SCORE_THRESHOLD      | 0.9                                  | Threshold for category score to raise an alert.                                      |
| WEBHOOK_URL          |                                      | URL to call in order to raise an alert.                                              |
