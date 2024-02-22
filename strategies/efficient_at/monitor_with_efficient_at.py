print('starting up (imports, initialization, etc.)')
import torch
from torchaudio import transforms
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels

import sounddevice as sd
import numpy as np
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Parameters
sampling_interval=float(os.environ['SAMPLING_INTERVAL'])
model_name=os.environ['MODEL_NAME']
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])
desired_sample_rate=32000 # ref: https://github.com/fschmid56/EfficientAT/issues/17#issuecomment-1681842622
buffer_size, remainder=divmod(desired_sample_rate, sampling_interval)
if remainder == 0:
  buffer_size=int(buffer_size)
else:
  raise ValueError(f'The sampling interval {sampling_interval}s is not a factor of the required sample rate of {desired_sample_rate}hz.')

# taken from default values in `inference.py`
window_size=800
hop_size=320
n_mels=128
strides=[2, 2, 2, 2]
head_type="mlp"

monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']

# Initialization
if model_name.startswith("dymn"):
  model = get_dymn(width_mult=NAME_TO_WIDTH(model_name),  pretrained_name=model_name, strides=strides)
else:
    model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name, strides=strides, head_type=head_type)
model.eval()

# model to preprocess waveform into mel spectrograms
mel = AugmentMelSTFT(n_mels=n_mels, sr=desired_sample_rate, win_length=window_size, hopsize=hop_size)
mel.eval()


# Define a function to resample audio
def resample(tensor):
  if recording_sample_rate != desired_sample_rate:
    resampler = transforms.Resample(orig_freq=recording_sample_rate, new_freq=desired_sample_rate)
    tensor = resampler(tensor)
  return tensor

# Inference loop
stream = sd.InputStream(device = 0, channels = 1, samplerate=recording_sample_rate, blocksize = buffer_size)
print('recording started')
# torch.no_grad is for performance https://github.com/microsoft/unilm/issues/998#issuecomment-1461310468
with stream, torch.no_grad():
  while True:
    # Record audio from the stream
    audio_array, _ = stream.read(buffer_size)

    # Crude method of converting ndarray(10000,1) to Tensor.shape(1,10000) with copilot
    tensor = torch.from_numpy(audio_array)
    tensor = torch.transpose(tensor, 0, 1)
    tensor = resample(tensor)

    spec = mel(tensor)
    preds, features = model(spec.unsqueeze(0))

    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    for k in range(5):
      label = labels[sorted_indexes[k]]
      prob = preds[sorted_indexes[k]]

      # # For debug output
      print(f'{label}: {prob:.3f}')
      if prob > score_threshold and label in monitored_categories:
        try:
          response = requests.post(webhook_url, json={label: prob})
          if response.status_code == 200:
            time.sleep(8)
        except:
          pass

    print("\n")
    time.sleep(sampling_interval)