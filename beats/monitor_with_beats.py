# Imports
print('starting up (imports, initialization, etc.)')
import torch
from BEATs import BEATs, BEATsConfig
import sounddevice as sd
import numpy as np
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Parameters
sampling_interval=float(os.environ['SAMPLING_INTERVAL'])
model_path=os.environ['MODEL_PATH']
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])
desired_sample_rate=16000 # specific to BEATS
buffer_size=15600
monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']

# Initialization
checkpoint = torch.load(model_path)
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()
audio_array = np.zeros((int(desired_sample_rate), 1))

def audio_callback(indata, frames, time, status):
  # Downsample the input data
  global audio_array
  audio_array = indata[::int(recording_sample_rate/desired_sample_rate)]


# Inference loop
stream = sd.InputStream(device = 0, channels = 1, samplerate=recording_sample_rate, callback = audio_callback, blocksize = buffer_size)
print('recording started')
with stream:
  while True:
    # predict the classification probability of each class
    audio_input_16khz = torch.tensor(audio_array)
    padding_mask = torch.zeros(1, 10000).bool()

    probs = BEATs_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]

    for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
      top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
      print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')

    time.sleep(sampling_interval)