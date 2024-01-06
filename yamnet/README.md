## Using YAMNET

Adapted from https://www.tensorflow.org/hub/tutorials/yamnet (parsing a wav file with tensorflow) and https://github.com/bwhitman/doorbell-detector (monitoring `sounddevice` with tensorflow)

Using `sounddevice` to extract samples, doing naive downsampling to the 16khz required for YAMNET inference, and finally using a [tflite-optimized](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) version.

### Running inference

1. `pip install -r requirements_yamnet.txt`
1. Fill out `.env`. Variable reference below. No quotes necessary.
1. `python3 monitor_with_yamnet.py`

| Variable name         | Example value  | Description                                                                                                                                                                                                                |
| --------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SAMPLING_INTERVAL     | 1              | Runs inference every second. YAMNET dataset consist of clips of 0.975s, so we try to match that.                                                                                                                           |
| RECORDING_SAMPLE_RATE | 32000          | The desired recording rate of your device. Ideally it should be a multiple of 16k, so it can be directly (naively) downsampled. Any other rate is fine if you are willing to write downsampling code (or import a library) |
| MODEL_PATH            | yamnet.tflite  | Path to the downloaded model                                                                                                                                                                                               |
| MONITORED_CATEGORIES  | Music,Doorbell | Labeled categories (separate by comma for multiple categories) to raise an alert for                                                                                                                                       |
| SCORE_THRESHOLD       | 0.5            | Threshold for category score to raise an alert. These are based off YAMNET's scores, so 0.5 serves a good starting point                                                                                                   |
| WEBHOOK_URL           |                | URL to call in order to raise an alert.                                                                                                                                                                                    |

## Fine-tuning YAMNET (unsolved)

Using a pre-trained model is easier. I tried to fine-tune it, but I ran into a few problems/mistakes:

- Binary classification task runs into the same problem as Teachable Machine approach
- Adding an additional layer/category on top of the existing categories was ideal, but because it is trained only on new data rather than together with the existing dataset, it is not very useful.
- Even so, converting the resulting model to tflite was unusable (different Keras shape error)

So for now I ended up using the pre-trained model as-is, and just monitored the most relevant category (some false positives expected).

## Reporting

I wanted to use local push notifications for zero latency. But it does not seem like a solved problem space yet. I ended up using Zapier for now.

Potential Options

- Progressive Web App (PWA) connecting to a local server using [Push API](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Tutorials/js13kGames/Re-engageable_Notifications_Push#push)
- Apple's [Local Push Connectivity](https://developer.apple.com/documentation/networkextension/local_push_connectivity)
- Zapier (Webhook with raw data + email with raw data content)
- Twilio SMS

## Running in the background

I just use `nohup`.

`nohup python3 monitor_with_yamnet.py &`
