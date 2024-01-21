print('starting up (imports, initialization, etc.)')
import torch
from torchaudio import transforms
import models
import pandas as pd

import sounddevice as sd
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Parameters
sampling_interval=float(os.environ['SAMPLING_INTERVAL'])
model_name=os.environ['MODEL_NAME']
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])
desired_sample_rate=16000 # ref: `inference.py``
buffer_size, remainder=divmod(desired_sample_rate, sampling_interval)
if remainder == 0:
  buffer_size=int(buffer_size)
else:
  raise ValueError(f'The sampling interval {sampling_interval}s is not a factor of the required sample rate of {desired_sample_rate}hz.')

# taken from default values in `inference.py`

monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']

# Initialization
model = getattr(models, model_name)(pretrained=True)
model.eval()

pd.read_csv('class_labels_indices.csv')
label_maps = pd.read_csv('class_labels_indices.csv').set_index('index')['display_name'].to_dict()

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

    zero_cache = None
    if 'SAT' in model_name:
      # First produce a "silence" cache
      *_, zero_cache = model(torch.zeros(1, int(model.cache_length / 100 * desired_sample_rate)), return_cache=True)

    if zero_cache is not None:
      output, zero_cache = model(tensor, cache=zero_cache, return_cache=True)
      output = output.squeeze(0)
    else:
      output = model(tensor).squeeze(0)

    # Print audio tagging top probabilities
    for k, (prob, label) in enumerate(zip(*output.topk(5))):
      lab_idx = label.item()
      label_name = label_maps[lab_idx]
      # # For debug output
      print(f'{label_name}: {prob:.3f}')
      if prob > score_threshold and label_name in monitored_categories:
        try:
          response = requests.post(webhook_url, json={label_name: prob})
          if response.status_code == 200:
            time.sleep(8)
        except:
          pass

    print("\n")
    time.sleep(sampling_interval)