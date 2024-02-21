print('starting up (imports, initialization, etc.)')
import torch
from torchaudio import transforms
# from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd

import sounddevice as sd
import time
import os
from dotenv import load_dotenv
load_dotenv()

from notifications.tapo import flicker
from notifications.webhook import ping

# Parameters
notification_method=os.environ['NOTIFICATION_METHOD']
sampling_interval=float(os.environ['SAMPLING_INTERVAL'])
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])
desired_sample_rate=16000 # ref: `inference.py``
buffer_size, remainder=divmod(desired_sample_rate, sampling_interval)
if remainder == 0:
  buffer_size=int(buffer_size)
else:
  raise ValueError(f'The sampling interval {sampling_interval}s is not a factor of the required sample rate of {desired_sample_rate}hz.')

# Common labels
monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']
label_maps = pd.read_csv('class_labels_indices.csv').set_index('index')['display_name'].to_dict()

# Choose ml method and model
ml_method = os.environ['ML_METHOD']
model_name = os.environ['MODEL_NAME']
from importlib import import_module
# import sys
# sys.path.append(os.getcwd() + '/methods/' + ml_method)
# all methods should implement the following functions
init, inference, format = import_module(f"methods.{ml_method}")

if model_name:
  print(f'initializing ml method {ml_method} with model {model_name}')
  model = init(model_name)
else:
  print(f'initializing ml method {ml_method}')
  model = init()

# Define a function to resample audio
def resample(tensor):
  if recording_sample_rate != desired_sample_rate:
    resampler = transforms.Resample(orig_freq=recording_sample_rate, new_freq=desired_sample_rate)
    tensor = resampler(tensor)
  return tensor

# Inference loop
iterationsRan = 0;
stream = sd.InputStream(device = 1, channels = 1, samplerate=recording_sample_rate, blocksize = buffer_size)
print('recording started')
# torch.no_grad is for performance https://github.com/microsoft/unilm/issues/998#issuecomment-1461310468
with stream, torch.no_grad(), """profile(
  activities=[ProfilerActivity.CPU],
  profile_memory=True,
) as prof""":
  # while iterationsRan < 10:
  while True:
    # Record audio from the stream
    audio_array, _ = stream.read(buffer_size)

    # Crude method of converting ndarray(10000,1) to Tensor.shape(1,10000) with copilot
    tensor = torch.from_numpy(audio_array)
    tensor = torch.transpose(tensor, 0, 1)
    tensor = resample(tensor)

    output = inference(tensor)

    # Print audio tagging top probabilities
    for (prob, label) in output:
      print(f'{label}: {prob:.3f}')
      if prob > score_threshold and label in monitored_categories:
        try:
          match notification_method:
            case 'webhook':
              ping({[label]: prob})
            case 'tapo':
              flicker()
        except:
          pass

    time.sleep(sampling_interval)
    # iterationsRan += 1

# prof.export_chrome_trace("trace.json")