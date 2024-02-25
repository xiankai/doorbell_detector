print('starting up (imports, initialization, etc.)')
import torch
from torchaudio import transforms

import sounddevice as sd
import time
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

from notifications.tapo import flicker
from notifications.webhook import ping
import asyncio

# Choose ml strategy and model
ml_strategy = os.environ['ML_STRATEGY']
model_name = os.environ['MODEL_NAME']
from importlib import import_module
import sys
sys.path.append(os.getcwd() + '/strategies/' + ml_strategy)
# all strategys should implement the following functions
module = import_module(f"strategies.{ml_strategy}.entrypoint")
init, inference, desired_sample_rate = module.init, module.inference, module.desired_sample_rate

# Parameters
notification_methods=os.environ['NOTIFICATION_METHODS'].split(',')
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])

sampling_interval = desired_sample_rate / recording_sample_rate
print(f'sampling every {sampling_interval} seconds')

# Common labels
monitored_categories=os.environ['MONITORED_CATEGORIES'].split(',')
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']

if model_name:
  print(f'initializing ml strategy {ml_strategy} with model {model_name}')
  model = init(model_name)
else:
  print(f'initializing ml strategy {ml_strategy}')
  model = init()

# Define a function to resample audio
def resample(tensor):
  if recording_sample_rate != desired_sample_rate:
    resampler = transforms.Resample(orig_freq=recording_sample_rate, new_freq=desired_sample_rate)
    tensor = resampler(tensor)
  return tensor


def check_schedule():
  now = datetime.datetime.now()
  if now.hour < 8:
    time.sleep((8 - now.hour - 1) * 3600 + (3600 - now.minute * 60))
    return
  if now.hour >= 8 and now.hour < 22:
    return
  if now.hour >= 22:
    time.sleep((8 + 24 - now.hour - 1) * 3600 + (3600 - now.minute * 60))
    return

# Inference loop
default_input_device = sd.default.device[0]

stream = sd.InputStream(device = default_input_device, channels = 1, samplerate=recording_sample_rate, blocksize=desired_sample_rate)
print('recording started')
# torch.no_grad is for performance https://github.com/microsoft/unilm/issues/998#issuecomment-1461310468
with stream, torch.no_grad():
  while True:
    check_schedule()

    # Record audio from the stream
    audio_array, _ = stream.read(desired_sample_rate)

    tensor = torch.from_numpy(audio_array)
    tensor = torch.transpose(tensor, 0, 1)

    output = inference(tensor)

    # print('\n')
    for (prob, label) in output:
      # print(f'{label}: {prob:.3f}')
      if prob > score_threshold and label in monitored_categories:
        try:
          if 'stdout' in notification_methods:
            print(f'{label}: {prob:.3f}')
            print('\n')

          if 'webhook' in notification_methods:
            asyncio.run(ping({[label]: prob}))
          elif 'tapo':
            asyncio.run(flicker())
        except:
          pass

    # time.sleep(sampling_interval)